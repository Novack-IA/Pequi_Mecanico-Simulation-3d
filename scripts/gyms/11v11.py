# In scripts/gyms/SelfPlay11v11.py

import os
import sys
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Add project root to Python path to solve the "No module named agent" error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agent.CEIA import Agent
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from scripts.commons.Script import Script

class SelfPlay11v11(gym.Env):
    """
    A multi-agent environment for 11 vs 11 self-play, designed for PPO.
    """
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw):
        super(SelfPlay11v11, self).__init__()
        self.step_counter = 0
        self.last_goals_scored = 0
        self.last_goals_conceded = 0
        self.last_ball_x = 0
        self.robot_type = r_type

        # --- Agent Setup ---
        self.learning_team = [Agent(ip, server_p + i, monitor_p + i, i + 1, self.robot_type, "Pequi", True, enable_draw, wait_for_server=False) for i in range(11)]
        self.frozen_opponent_team = [Agent(ip, server_p + 11 + i, monitor_p + 11 + i, i + 1, "Opponent", True, enable_draw, wait_for_server=False) for i in range(11)]
        self.opponent_model = None

        # --- Observation and Action Spaces ---
        obs_size_per_player = 50
        act_size_per_player = 22
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11 * obs_size_per_player,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(11 * act_size_per_player,), dtype=np.float32)

    def load_opponent_model(self, model_path):
        if os.path.exists(model_path):
            self.opponent_model = PPO.load(model_path, env=self)
            print(f"Opponent model loaded from {model_path}")
        else:
            print("No opponent model found. Opponent will act randomly.")

    def step(self, combined_action):
        actions = np.split(combined_action, 11)
        for i, player in enumerate(self.learning_team):
            player.world.robot.set_joints_target_position_direct(slice(2, 24), actions[i] * 25, harmonize=False)
            player.scom.commit_and_send(player.world.robot.get_command())

        if self.opponent_model:
            opponent_obs = self._get_opponent_observation()
            opponent_actions, _ = self.opponent_model.predict(opponent_obs, deterministic=True)
            opponent_actions = np.split(opponent_actions, 11)
            for i, player in enumerate(self.frozen_opponent_team):
                player.world.robot.set_joints_target_position_direct(slice(2, 24), opponent_actions[i] * 25, harmonize=False)
                player.scom.commit_and_send(player.world.robot.get_command())

        for player in self.learning_team: player.scom.receive()
        for player in self.frozen_opponent_team: player.scom.receive()

        self.step_counter += 1
        obs = self._get_observation()
        reward = self._calculate_aggressive_reward()
        done = self._check_if_done()
        return obs, reward, done, {}

    def reset(self):
        self.step_counter = 0
        self.last_ball_x = 0
        self.last_goals_scored = 0
        self.last_goals_conceded = 0
        
        for player in self.learning_team: player.beam()
        for player in self.frozen_opponent_team: player.beam()
        
        self.learning_team[0].scom.unofficial_move_ball((0, 0, 0.042))
        return self._get_observation()

    def _get_observation(self):
        all_obs = []
        for player in self.learning_team:
            obs = np.zeros(50, dtype=np.float32)
            r, w = player.world.robot, player.world
            obs[0:3] = r.get_head_abs_vel(2) / 5.0
            obs[40:43] = np.array(w.ball_rel_torso_cart_pos)
            all_obs.append(obs)
        return np.concatenate(all_obs)

    def _get_opponent_observation(self):
        all_obs = []
        for player in self.frozen_opponent_team:
            obs = np.zeros(50, dtype=np.float32)
            r, w = player.world.robot, player.world
            obs[0:3] = r.get_head_abs_vel(2) / 5.0
            obs[40:43] = np.array(w.ball_rel_torso_cart_pos)
            all_obs.append(obs)
        return np.concatenate(all_obs)

    def _calculate_aggressive_reward(self):
        w = self.learning_team[0].world
        ball_pos = w.ball_abs_pos
        reward = 0

        ball_progress = ball_pos[0] - self.last_ball_x
        reward += ball_progress * 10
        self.last_ball_x = ball_pos[0]

        if ball_pos[0] > 13.2: reward += 0.5
        if w.goals_scored > self.last_goals_scored: reward += 10
        if w.goals_conceded > self.last_goals_conceded: reward -= 10
        
        self.last_goals_scored = w.goals_scored
        self.last_goals_conceded = w.goals_conceded
        return reward

    def _check_if_done(self):
        return self.learning_team[0].world.play_mode != 'PlayOn' or self.step_counter > 6000

    def render(self, mode='human'): pass
    def close(self):
        for p in self.learning_team: p.terminate()
        for p in self.frozen_opponent_team: p.terminate()


class Train(Train_Base):
    def __init__(self, script):
        super().__init__(script)

    def train(self, args):
        n_envs = 1 
        total_steps = 100_000_000
        n_steps = 8192
        generations_before_update = 50
        model_path = f'./scripts/gyms/logs/SelfPlay11v11_R{self.robot_type}/'
        os.makedirs(model_path, exist_ok=True)

        print(f"Starting training with {n_envs} environment(s).")
        print("Model Path:", model_path)

        servers = Server(self.server_p, self.monitor_p_1000, n_envs * 22)

        def init_env(i_env):
            def thunk():
                return SelfPlay11v11(self.ip, self.server_p + i_env * 22, self.monitor_p_1000 + i_env * 22, self.robot_type, False)
            return thunk

        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        model = PPO("MlpPolicy", env, verbose=1, n_steps=n_steps, device="cpu")

        try:
            for gen in range(total_steps // (n_steps * n_envs * generations_before_update)):
                print(f"--- Generation {gen + 1} ---")
                model.learn(total_timesteps=n_steps * n_envs * generations_before_update, reset_num_timesteps=False)
                current_model_path = f"{model_path}/model_gen_{gen + 1}.zip"
                model.save(current_model_path)
                env.env_method("load_opponent_model", model_path=current_model_path)
        except (KeyboardInterrupt, Exception) as e:
            print(f"\nException occurred: {e}. Saving model and aborting...")
            model.save(f"{model_path}/dribble_agent_aborted.zip")
        finally:
            print("Closing environments and servers...")
            env.close()
            servers.kill()
            print("Training complete.")

# This makes the script executable
if __name__ == "__main__":
    script = Script()
    trainer = Train(script)
    trainer.train({})