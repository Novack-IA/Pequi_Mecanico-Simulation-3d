# In Run_SelfPlay_Training.py (or aaaa.py)

import os
import sys
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# This is crucial for making imports work correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agent.CEIA import Agent
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from scripts.commons.Script import Script

class SelfPlay11v11Env(gym.Env):
    """
    A multi-agent environment for 11 vs 11 self-play.
    """
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw):
        super(SelfPlay11v11Env, self).__init__()
        self.step_counter = 0
        self.last_ball_x = 0
        self.last_goals_scored = 0
        self.last_goals_conceded = 0
        self.robot_type = r_type

        self.learning_team = [Agent(ip, server_p + i, monitor_p + i, i + 1, "Pequi", enable_draw, enable_draw, wait_for_server=False) for i in range(11)]
        self.opponent_team = [Agent(ip, server_p + 11 + i, monitor_p + 11 + i, i + 1, "Opponent", enable_draw, enable_draw, wait_for_server=False) for i in range(11)]
        self.opponent_model = None

        obs_size_per_player = 50
        # --- CORRECTED ACTION SIZE ---
        # The robot has 22 joints, but we control joints 2 through 21 (20 total).
        act_size_per_player = 20
        # ---------------------------
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11 * obs_size_per_player,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(11 * act_size_per_player,), dtype=np.float32)

    def load_opponent_model(self, model_path):
        if os.path.exists(model_path):
            self.opponent_model = PPO.load(model_path, env=self)
            print(f"Opponent model loaded from {model_path}")
        else:
            print("No opponent model found. Opponent will not act.")

    def step(self, combined_action):
        learning_actions = np.split(combined_action, 11)
        for i, player in enumerate(self.learning_team):
            # --- CORRECTED FUNCTION CALL ---
            # Using the correct slice and setting harmonize to False
            player.world.robot.set_joints_target_position_direct(slice(2, player.world.robot.no_of_joints), learning_actions[i], harmonize=False)
            player.scom.commit_and_send(player.world.robot.get_command())
            # ---------------------------

        if self.opponent_model:
            opponent_obs = self._get_opponent_observation()
            opponent_actions, _ = self.opponent_model.predict(opponent_obs, deterministic=True)
            opponent_actions_split = np.split(opponent_actions, 11)
            for i, player in enumerate(self.opponent_team):
                player.world.robot.set_joints_target_position_direct(slice(2, player.world.robot.no_of_joints), opponent_actions_split[i], harmonize=False)
                player.scom.commit_and_send(player.world.robot.get_command())
        else:
            for player in self.opponent_team: player.scom.commit_and_send(b'')

        for player in self.learning_team: player.scom.receive(update=True)
        for player in self.opponent_team: player.scom.receive(update=True)

        self.step_counter += 1
        obs = self._get_observation()
        reward = self._calculate_aggressive_reward()
        done = self._check_if_done()
        
        return obs, reward, done, {}

    def reset(self):
        self.step_counter = 0
        self.last_ball_x = 0
        self.last_goals_scored = self.learning_team[0].world.goals_scored
        self.last_goals_conceded = self.learning_team[0].world.goals_conceded
        
        for player in self.learning_team: player.beam()
        for player in self.opponent_team: player.beam()
        
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
        for player in self.opponent_team:
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
        
        return reward

    def _check_if_done(self):
        return self.learning_team[0].world.play_mode != 'PlayOn' or self.step_counter > 6000

    def render(self, mode='human'): pass
    def close(self):
        for p in self.learning_team: p.terminate()
        for p in self.opponent_team: p.terminate()


class SelfPlayTrainer(Train_Base):
    def __init__(self, script):
        super().__init__(script)

    def train(self):
        n_envs = 1 
        total_steps = 100_000_000
        n_steps = 8192
        generations_before_update = 50
        model_dir = f'./scripts/gyms/logs/SelfPlay11v11_R{self.robot_type}/'
        os.makedirs(model_dir, exist_ok=True)

        print(f"Starting training with {n_envs} environment(s).")
        print("Model Path:", model_dir)
        
        servers = Server(self.server_p, self.monitor_p_1000, n_envs * 22)

        def init_env(i_env):
            def thunk():
                return SelfPlay11v11Env(self.ip, self.server_p + i_env * 22, self.monitor_p_1000 + i_env * 22, self.robot_type, False)
            return thunk

        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        
        latest_model = self.get_latest_model(model_dir)
        if latest_model:
            print(f"Loading latest model: {latest_model}")
            model = PPO.load(latest_model, env=env, device="cpu")
        else:
            print("Creating a new PPO model.")
            model = PPO("MlpPolicy", env, verbose=1, n_steps=n_steps, device="cpu")

        try:
            for gen in range(total_steps // (n_steps * n_envs * generations_before_update)):
                print(f"--- Generation {gen + 1} ---")
                model.learn(total_timesteps=n_steps * n_envs * generations_before_update, reset_num_timesteps=False)
                
                current_model_path = f"{model_dir}/model_gen_{gen + 1}.zip"
                model.save(current_model_path)
                
                env.env_method("load_opponent_model", model_path=current_model_path)

        except (KeyboardInterrupt, Exception) as e:
            print(f"\nException occurred: {e}. Saving model and aborting...")
            model.save(f"{model_dir}/aborted_model.zip")
        finally:
            print("Closing environments and servers...")
            env.close()
            servers.kill()
            print("Training complete.")

    def get_latest_model(self, model_dir):
        files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith("model_gen_") and f.endswith(".zip")]
        if not files: return None
        return max(files, key=os.path.getmtime)

if __name__ == "__main__":
    script = Script(cpp_builder_unum=1)
    trainer = SelfPlayTrainer(script)
    trainer.train()