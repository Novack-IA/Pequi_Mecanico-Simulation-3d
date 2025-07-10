import os
import argparse
import multiprocessing
import time
import subprocess
import signal

# --- RL & Environment Imports ---
# Make sure you have these packages installed:
# pip install stable-baselines3[extra] gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
except ImportError as e:
    print(f"RL library import error: {e}")
    print("Please run 'pip install stable-baselines3[extra] gymnasium' to install required packages.")
    # We don't exit here, as 'play' mode might still work.

# --- Agent and Script Imports ---
# Assuming these are in the correct path.
try:
    from scripts.commons.Script import Script
    from agent.CEIA import Agent as NormalAgent
    from agent.CEIA_Penalty import Agent as PenaltyAgent
except ImportError as e:
    print(f"Warning: Could not import agent modules: {e}")
    print("Training mode will not work. 'play' mode might fail if it relies on these.")
    NormalAgent = None
    PenaltyAgent = None


# ==============================================================================
# SECTION 1: REINFORCEMENT LEARNING ENVIRONMENT (for --mode train)
# ==============================================================================

class SoccerEnv(gym.Env):
    """
    A custom Gym environment for RoboCup 3D Soccer Simulation (SimSpark).

    This environment manages two full teams of 11 players for self-play training.
    It handles starting/stopping the SimSpark server and the player agents.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, host='localhost', port=3100, team_name_learner='Learner', team_name_opponent='Opponent', opponent_model_path=None):
        super(SoccerEnv, self).__init__()

        self.host = host
        self.port = port
        self.team_name_learner = team_name_learner
        self.team_name_opponent = team_name_opponent
        self.simspark_process = None
        self.agents = [] # Will hold all 22 agent instances
        self.opponent_model_path = opponent_model_path
        self.opponent_policy = None

        # TODO: Define your action and observation space.
        # This is CRITICAL and depends entirely on your agent's implementation.

        # --- ACTION SPACE ---
        # The action space defines what the PPO model can "do".
        self.n_players_per_team = 11
        n_actions_per_player = 5 # Placeholder value
        self.action_space = spaces.MultiDiscrete([n_actions_per_player] * self.n_players_per_team)

        # --- OBSERVATION SPACE ---
        # The observation space defines what the PPO model "sees".
        obs_dims_per_player = 20 # Placeholder value
        total_obs_dims = obs_dims_per_player * self.n_players_per_team
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dims,), dtype=np.float32)

    def _start_simulator(self):
        """Starts the SimSpark server as a subprocess."""
        print("Starting SimSpark server...")
        try:
            self.simspark_process = subprocess.Popen(['simspark'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5) # Give the server time to initialize
            print(f"SimSpark started with PID: {self.simspark_process.pid}")
        except FileNotFoundError:
            print("ERROR: 'simspark' command not found.")
            print("Please ensure SimSpark is installed and in your system's PATH.")
            raise

    def _start_agents(self):
        """Initializes and connects all 22 player agents."""
        self.agents = []
        print("Starting agents...")
        for i in range(1, self.n_players_per_team + 1):
            # Learner team agents are controlled by the PPO model, so they don't need their own logic here.
            # We still need an object to interface with the server for them.
            learner_agent = NormalAgent(self.host, self.port, None, i, self.team_name_learner,
                                        False, False, False, False)
            self.agents.append(learner_agent)
            
            # Opponent team agents will use their own logic or a loaded model.
            opponent_agent = NormalAgent(self.host, self.port, None, i, self.team_name_opponent,
                                         False, False, False, False)
            self.agents.append(opponent_agent)
        print(f"{len(self.agents)} agents created.")
        time.sleep(2) # Allow agents to connect

    def set_opponent_model(self, model_path):
        """Loads the policy for the opponent team."""
        print(f"Loading opponent model from: {model_path}")
        self.opponent_model_path = model_path
        # We load the policy here to be used in the step function.
        self.opponent_policy = PPO.load(self.opponent_model_path, env=self).policy

    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.
        This involves restarting the simulator and all agents.
        """
        super().reset(seed=seed)

        if self.simspark_process:
            print(f"Stopping SimSpark server (PID: {self.simspark_process.pid})...")
            self.simspark_process.terminate()
            self.simspark_process.wait()

        self._start_simulator()
        self._start_agents()

        initial_observation = self._get_observation()
        info = {}
        return initial_observation, info

    def _get_observation(self, team_agents):
        """
        Collects state information from a given team's agents and formats it.
        """
        # TODO: Implement this logic.
        # This is a placeholder implementation.
        # You need to get the actual state from your agents.
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _calculate_reward(self):
        """
        Calculates the reward for the current step.
        """
        # TODO: Implement your reward function.
        return 0.0

    def _is_terminated(self):
        """
        Checks if the episode has ended.
        """
        # TODO: Implement termination logic.
        return False

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        # 1. Map actions from the PPO model to learner agent commands
        # TODO: Implement the mapping from the `action` array to specific commands.
        learner_agents = self.agents[::2]
        # for i, player_action in enumerate(action):
        #     agent = learner_agents[i]
        #     # agent.execute_action(player_action) # You'll need a method like this

        # 2. Let the opponent team act
        opponent_agents = self.agents[1::2]
        if self.opponent_policy:
            # Get opponent observation
            opponent_obs = self._get_observation(opponent_agents)
            # Predict actions with the loaded opponent model
            opponent_actions, _ = self.opponent_policy.predict(opponent_obs, deterministic=True)
            # TODO: Map opponent_actions to opponent agent commands
            # for i, player_action in enumerate(opponent_actions):
            #     agent = opponent_agents[i]
            #     # agent.execute_action(player_action)
        else:
            # Fallback to original logic if no model is loaded
            for agent in opponent_agents:
                agent.think_and_send()

        # 3. Let the simulation run for a cycle
        for agent in self.agents:
            agent.scom.receive()

        # 4. Get results for the learner team
        observation = self._get_observation(learner_agents)
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        """Cleans up the environment."""
        print("Closing environment and stopping SimSpark.")
        if self.simspark_process:
            self.simspark_process.terminate()
            self.simspark_process.wait()


# ==============================================================================
# SECTION 2: PLAYER LAUNCHER (for --mode play)
# ==============================================================================

def run_player(host, port, monitor_port, unum, team_name, is_penalty_shootout, is_debug_mode, wait_for_server, is_magma_proxy):
    """
    This function contains the logic for a single player agent.
    """
    try:
        print(f"[Player {unum}]: Initializing...")
        AgentClass = PenaltyAgent if is_penalty_shootout else NormalAgent
        # Calling with positional arguments as the Agent class expects
        player = AgentClass(host, port, monitor_port if is_debug_mode else None, unum, team_name,
                            is_debug_mode, is_debug_mode,
                            wait_for_server, is_magma_proxy)
        print(f"[Player {unum}]: Running main loop...")
        while True:
            player.think_and_send()
            player.scom.receive()
    except KeyboardInterrupt:
        print(f"[Player {unum}]: Process interrupted. Exiting.")
    except Exception as e:
        print(f"[Player {unum}]: An error occurred: {e}")


# ==============================================================================
# SECTION 3: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="Launcher and Trainer for the CEIA RoboCup team.")
    parser.add_argument("--mode", type=str, default="play", choices=["train", "play"], help="Mode to run the script in.")
    # Common arguments
    parser.add_argument("-i", "--host", default="localhost", help="The server IP address.")
    parser.add_argument("-p", "--port", type=int, default=3100, help="The server port for agents.")
    # Play mode arguments
    parser.add_argument("-m", "--monitor-port", type=int, default=3200, help="The server port for the monitor.")
    parser.add_argument("-t", "--team-name", default="Pequi-Mecanico", help="The name of the team for play mode.")
    parser.add_argument("-P", "--penalty", action='store_true', help="Enable penalty shootout mode.")
    parser.add_argument("-D", "--debug", action='store_true', help="Enable debug mode (logging and drawing).")
    parser.add_argument("--wait-for-server", action='store_true', help="Agent waits for server before starting.")
    parser.add_argument("--magma-proxy", action='store_true', help="Enable magmaFatProxy.")
    # Train mode arguments
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total timesteps for training.")
    parser.add_argument("--update-epochs", type=int, default=10, help="Number of training epochs before updating the opponent model.")


    args = parser.parse_args()

    # --- TRAIN MODE ---
    if args.mode == 'train':
        if NormalAgent is None:
            print("Cannot run train mode: Agent classes failed to import.")
            exit(1)

        print("--- Starting Training Mode ---")
        models_dir = "models/PPO_selfplay"
        logdir = "logs"
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)

        env = SoccerEnv(host=args.host, port=args.port)
        
        learner_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

        opponent_model_path = f"{models_dir}/opponent_0.zip"
        learner_model.save(opponent_model_path)
        env.set_opponent_model(opponent_model_path)

        timesteps_per_epoch = args.total_timesteps // args.update_epochs
        for epoch in range(args.update_epochs):
            print(f"--- Epoch {epoch + 1} / {args.update_epochs} ---")
            learner_model.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False, tb_log_name="PPO_SelfPlay")

            learner_model_path = f"{models_dir}/learner_epoch_{epoch+1}.zip"
            learner_model.save(learner_model_path)
            print(f"Learner model saved to {learner_model_path}")

            env.set_opponent_model(learner_model_path)
            print("Opponent model has been updated to the latest learner version for the next epoch.")

        env.close()

    # --- PLAY MODE ---
    elif args.mode == 'play':
        if NormalAgent is None:
            print("Cannot run play mode: Agent classes failed to import.")
            exit(1)

        print("--- Starting Play Mode ---")
        print(f"Host: {args.host}, Port: {args.port}, Team: {args.team_name}")
        processes = []
        num_players = 11
        for i in range(1, num_players + 1):
            process = multiprocessing.Process(
                target=run_player,
                args=(args.host, args.port, args.monitor_port, i, args.team_name,
                      args.penalty, args.debug, args.wait_for_server, args.magma_proxy)
            )
            processes.append(process)
            process.start()
            print(f"Launched player {i}...")
            time.sleep(0.1)

        print(f"\nAll {num_players} players launched. Press Ctrl+C to stop.")
        try:
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            print("\n--- Shutting Down ---")
            for process in processes:
                process.terminate()
                process.join()
            print("All player processes terminated.")

