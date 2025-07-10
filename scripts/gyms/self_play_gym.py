import os
import subprocess
import time
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Assuming these are your custom project imports
from scripts.commons.Script import Script
from agent.Base_AgentCEIA import Base_Agent as Agent
from math_ops.Math_Ops import Math_Ops as M

# FIX: Define constants for better configuration and clarity.
# SimSpark's field is roughly 30x20 meters. A larger boundary handles objects outside the field.
FIELD_BOUNDARY = 32.0 
RCSSSERVER_EXECUTABLE = "rcssserver3d" # Path to your SimSpark server executable

class SelfPlayEnv(gym.Env):
    """
    Custom Gym environment for 11-vs-11 self-play training.
    FIXED:
    - Observation space now includes the ball and is normalized.
    - step() method correctly uses actions from the policy.
    - reset() method correctly resets the ball.
    - close() method no longer manages the server process.
    """
    def __init__(self, script: Script, opponent_model_path: str = None):
        super(SelfPlayEnv, self).__init__()
        self.script = script
        self.a = self.script.args

        # Create 11 players for the learning team (Home)
        self.script.batch_create(Agent, ((self.a.i, self.a.p, self.a.m, u + 1, self.a.r, "Home") for u in range(11)))
        self.home_players = self.script.players[-22:-11] # More robust slicing

        # Create 11 players for the opponent team (Away)
        self.script.batch_create(Agent, ((self.a.i, self.a.p, self.a.m, u + 1, self.a.r, "Away") for u in range(11)))
        self.away_players = self.script.players[-11:]

        self.players = self.home_players + self.away_players
        
        # Load the opponent's model, if provided
        self.opponent_model = None
        if opponent_model_path and os.path.exists(opponent_model_path):
            print(f"Loading opponent model from: {opponent_model_path}")
            self.opponent_model = PPO.load(opponent_model_path)
        else:
            print("No opponent model found or specified. Opponent will use default behavior.")

        # --- FIX: Define a more robust Action and Observation Space ---
        # Action space: 5 discrete actions for each of the 11 players.
        # This is a "centralized controller" setup.
        # Note: The action space size is 5^11, which is huge. MARL is often a better approach.
        # 0: Move to Ball, 1: Kick to Goal, 2: Dribble towards Goal, 3: Pass to nearest teammate, 4: Hold Position
        self.action_space = gym.spaces.MultiDiscrete([5] * 11)

        # Observation space: Contains normalized (x, y) positions for the ball and all 22 players.
        # Shape is ((1_ball + 22_players) * 2_coords) = 46.
        obs_shape = ((1 + 22) * 2,)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=obs_shape, dtype=np.float32)

        # Allow a moment for agents to connect to the server before the first reset
        print("Waiting for agents to connect...")
        time.sleep(3)
        self.reset()

    def _execute_team_action(self, team_players: list, team_action: np.ndarray):
        """Helper function to map action IDs to agent behaviors for a team."""
        for i, player in enumerate(team_players):
            action_id = team_action[i]
            
            # --- FIX: This logic maps the model's output (action_id) to actual behaviors ---
            # You MUST adapt this section to match the methods available in your 'Agent.behavior' class.
            world = player.world
            robot_pos = world.robot.position[:2]
            ball_pos = world.ball.position[:2]
            
            # Determine goal position based on the team
            is_home_team = (player.team_name == "Home")
            goal_pos = (15, 0) if is_home_team else (-15, 0)

            # Check if player is near the ball, a common precondition for kicking/dribbling
            is_near_ball = M.distance(robot_pos, ball_pos) < 0.8

            if action_id == 0: # Move to Ball
                player.behavior.execute("Basic_Move", *ball_pos)
            elif action_id == 1: # Kick to Goal
                if is_near_ball:
                    player.behavior.execute("Basic_Kick", M.vector_angle(goal_pos - ball_pos), 100) # 100% power
                else: # If not near ball, move to it
                    player.behavior.execute("Basic_Move", *ball_pos)
            elif action_id == 2: # Dribble
                if is_near_ball:
                    dribble_target = robot_pos + M.unit_vector(goal_pos - robot_pos) * 5
                    player.behavior.execute("Dribble", *dribble_target)
                else:
                    player.behavior.execute("Basic_Move", *ball_pos)
            elif action_id == 3: # Short Pass
                if is_near_ball:
                    teammates = [p for p in team_players if p.unum != player.unum]
                    if teammates:
                        # Find the nearest teammate to pass to
                        nearest_teammate = min(teammates, key=lambda t: M.distance(robot_pos, t.world.robot.position[:2]))
                        player.behavior.execute("Basic_Kick_To", *nearest_teammate.world.robot.position[:2], 30) # 30% power
                    else: # No teammates, just kick to goal
                        player.behavior.execute("Basic_Kick", M.vector_angle(goal_pos - ball_pos), 100)
                else:
                    player.behavior.execute("Basic_Move", *ball_pos)
            else: # action_id == 4: Hold Position
                player.behavior.execute("Do_Nothing") # Assumes a null/holding behavior exists

    def step(self, action: np.ndarray):
        # 1. Execute action for the learning team (Home)
        self._execute_team_action(self.home_players, action)

        # 2. Get and execute action for the opponent team (Away)
        if self.opponent_model:
            # The opponent's observation should be from its perspective (mirrored)
            # For simplicity, we use the same obs here, but mirroring is recommended for true self-play
            obs = self._get_observation() 
            away_action, _ = self.opponent_model.predict(obs, deterministic=True)
            self._execute_team_action(self.away_players, away_action)
        else:
            # Default behavior if no opponent model (e.g., first generation)
            default_away_action = np.random.randint(0, 5, 11) # Random actions
            self._execute_team_action(self.away_players, default_away_action)

        # 3. Advance the simulation
        self.script.batch_commit_and_send()
        self.script.batch_receive()

        # 4. Get results
        observation = self._get_observation()
        reward, done = self._calculate_reward()
        info = {}
        
        return observation, reward, done, info

    def reset(self):
        # --- FIX: Reset the ball's position in addition to players ---
        # Use a single agent to send server commands
        commander = self.players[0].scom
        
        # Beam players and ball to kickoff positions
        commander.unofficial_kickoff()
        # Ensure the game mode is set to PlayOn to start immediately
        commander.unofficial_set_play_mode("PlayOn")
        
        # Run one simulation step for beamed positions to be updated in the world model
        self.script.batch_commit_and_send()
        self.script.batch_receive()
        time.sleep(0.1) # A small delay to ensure the world state is updated
        
        return self._get_observation()

    def _get_observation(self):
        # --- FIX: Include ball position and normalize the entire observation ---
        world = self.home_players[0].world
        
        ball_pos = world.ball.position[:2]
        
        obs_list = [ball_pos]
        for p in self.home_players:
            obs_list.append(p.world.robot.position[:2])
        for p in self.away_players:
            obs_list.append(p.world.robot.position[:2])
        
        # Flatten and normalize the list of 2D positions
        obs_flat = np.array(obs_list, dtype=np.float32).flatten()
        obs_normalized = obs_flat / FIELD_BOUNDARY
        
        return np.clip(obs_normalized, -1.0, 1.0)

    def _calculate_reward(self):
        reward = 0
        done = False
        
        w_home = self.home_players[0].world
        game_mode = w_home.game.play_mode
        
        # Major events that end the episode
        if game_mode == w_home.M_GOAL_OUR:
            reward = 10.0
            done = True
        elif game_mode == w_home.M_GOAL_THEIR:
            reward = -10.0
            done = True
        elif game_mode not in [w_home.M_PLAY_ON, w_home.M_KICK_OFF_OUR, w_home.M_KICK_OFF_THEIR]:
            # Any other mode (out of bounds, etc.) ends the episode
            done = True

        # Shaping rewards (only if episode is not done)
        if not done:
            # Reward for ball possession
            if w_home.ball.last_touch_side == 'left': # 'left' is Home team
                reward += 0.1
            
            # Reward for the ball being in the opponent's half
            if w_home.ball.position[0] > 0:
                reward += 0.05

        return reward, done

    def render(self, mode='human'):
        pass # Visualization is handled by RoboViz

    def close(self):
        # --- FIX: Do not kill the server here. Let the main script handle it. ---
        print("Closing SelfPlayEnv.")
        pass

class Train:
    """Helper class to orchestrate the training process for a single generation."""
    def __init__(self, script: Script):
        self.script = script

    def train(self, total_timesteps: int, opponent_model_path: str = None, save_path_prefix: str = "gen"):
        
        print("Creating training environment...")
        # FIX: We only need ONE environment instance for both training and evaluation.
        train_env = SelfPlayEnv(self.script, opponent_model_path)
        
        # FIX: The 'eval_env' is no longer created.
        # print("Creating evaluation environment...")
        # eval_env = SelfPlayEnv(self.script, opponent_model_path)
        
        save_path = f"./models/{save_path_prefix}"
        os.makedirs(save_path, exist_ok=True)
        
        model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=f"./logs/{save_path_prefix}/")
        
        # --- FIX: Pass the 'train_env' to the callback ---
        # This tells the callback to run evaluations in the same environment,
        # avoiding the creation of new, conflicting agents.
        eval_callback = EvalCallback(train_env, best_model_save_path=save_path,
                                     log_path=save_path, eval_freq=10000,
                                     deterministic=True, render=False)

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save(f"{save_path}/final_model.zip")

        print(f"Training complete. Best model saved in: {save_path}/best_model.zip")
        
        # Now we only need to close one environment.
        train_env.close()

def main():
    """Main script for training across N generations."""
    generations = 100
    total_timesteps_per_generation = 100000
    last_best_model = None
    
    # This single Script instance is passed to all components.
    script = Script()

    for gen in range(1, generations + 1):
        print("\n" + "="*45)
        print(f" STARTING TRAINING GENERATION {gen}/{generations} ")
        print("="*45)

        # --- FIX: Start and stop the SimSpark server for each generation ---
        server_process = None
        try:
            print("Starting rcssserver3d...")
            server_process = subprocess.Popen(RCSSSERVER_EXECUTABLE, shell=False)
            # Give the server a moment to initialize
            time.sleep(10)

            trainer = Train(script)
            trainer.train(
                total_timesteps=total_timesteps_per_generation,
                opponent_model_path=last_best_model,
                save_path_prefix=f"gen_{gen}"
            )

            # Update the path for the next generation's opponent
            last_best_model = f"./models/gen_{gen}/best_model.zip"

            if not os.path.exists(last_best_model):
                print(f"WARNING: Best model {last_best_model} not found. Next gen will use default opponent.")
                last_best_model = None
        
        except Exception as e:
            print(f"An error occurred during generation {gen}: {e}")
            print("Stopping training.")
            break
        
        finally:
            if server_process:
                print("Terminating rcssserver3d...")
                server_process.terminate()
                server_process.wait()
        
        print(f"\nEND OF GENERATION {gen}.\n")

if __name__ == "__main__":
    main()