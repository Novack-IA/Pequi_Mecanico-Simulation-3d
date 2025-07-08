# Salve este arquivo como: scripts/gyms/Strong_Kick_Gym.py

from agent.Base_AgentCEIA import Base_Agent as Agent
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os, gym
import numpy as np
from math_ops.Math_Ops import Math_Ops as M

'''
Objetivo:
Treinar o Robô Tipo 4 para executar o chute mais FORTE e PRECISO possível
em direção ao gol a partir de uma posição preparada.
----------
- class Strong_Kick_Gym: implementa um ginásio personalizado do OpenAI.
- class Train: implementa os algoritmos para treinar um novo modelo.
'''

class Strong_Kick_Gym(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
        if r_type != 4:
            print("AVISO: Este ambiente de treino é otimizado para o Robô Tipo 4, que possui articulações nos dedos.")

        self.robot_type = r_type
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0

        # --- Espaço de Observação (42) ---
        # Focado no estado das pernas e na relação com a bola
        obs_size = 42
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # --- Espaço de Ação (12) ---
        # A IA controlará as 12 articulações das pernas e pés (llj1 a rlj6)
        MAX = np.finfo(np.float32).max
        act_size = 12 
        self.action_space = gym.spaces.Box(low=-MAX, high=MAX, shape=(act_size,), dtype=np.float32)

    def observe(self):
        r = self.player.world.robot
        w = self.player.world
        
        obs = np.zeros(42, dtype=np.float32)

        # Estado do Robô (6)
        obs[0:3]   = r.imu_torso_orientation / 180.0
        obs[3:6]   = r.gyro / 200.0

        # Estado das Articulações (24)
        obs[6:18]  = r.joints_position[2:14] / 180.0
        obs[18:30] = r.joints_speed[2:14] / 7.0

        # Estado da Bola (5)
        ball_rel_pos = w.ball_rel_torso_cart_pos
        obs[30:33] = ball_rel_pos
        obs[33]    = np.linalg.norm(ball_rel_pos)
        obs[34]    = M.target_abs_angle(r.loc_head_position[:2], (15.05, 0)) - r.imu_torso_orientation

        return obs

    def sync(self):
        self.player.scom.commit_and_send(self.player.world.robot.get_command())
        self.player.scom.receive()

    def reset(self):
        self.step_counter = 0
        r = self.player.world.robot
        
        # Posição ideal para o chute. O robô já começa preparado.
        self.player.scom.unofficial_beam((13.8, 0, r.beam_height), 180)
        self.player.scom.unofficial_move_ball((14.0, -0.09, 0.042)) # Bola ligeiramente à direita do robô
        
        # O robô executa a primeira fase do chute do XML para se posicionar
        self.player.behavior.execute_to_completion("Kick_Motion_strong")

        return self.observe()

    def step(self, action):
        r = self.player.world.robot
        w = self.player.world
        
        # Ação: Define diretamente a posição alvo das 12 articulações da perna
        r.set_joints_target_position_direct(slice(2, 14), action * 30, harmonize=False)

        self.sync()
        self.step_counter += 1
        
        ball_pos = w.ball_abs_pos
        ball_vel = w.get_ball_abs_vel(1)
        
        # --- Função de Recompensa Otimizada para FORÇA e PRECISÃO ---
        
        # 1. Recompensa principal: Velocidade da bola em direção ao gol
        reward = ball_vel[0] * 5.0

        # 2. Penalidade por desvio lateral (incentiva a precisão)
        reward -= abs(ball_vel[1]) * 3.0

        # 3. Penalidade por chutar para cima (incentiva chutes rasteiros e fortes)
        reward -= abs(ball_vel[2]) * 2.0
        
        terminal = False
        if ball_pos[0] > 15.05 and abs(ball_pos[1]) < 1.05:
            reward += 100 # GOL! Recompensa máxima.
            terminal = True
        elif r.loc_head_z < 0.3: # Penalidade por cair
            reward = -30 
            terminal = True
        elif self.step_counter > 150 or ball_vel[0] < -0.2: # Fim por tempo ou chute para trás
            terminal = True

        return self.observe(), reward, terminal, {}
    
    def render(self, mode='human', close=False): return
    def close(self):
        Draw.clear_all()
        self.player.terminate()

class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):
        n_envs = min(12, os.cpu_count())
        n_steps_per_env = 1024
        total_steps = 25_000_000
        folder_name = f'Strong_Kick_Gym_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Caminho do Modelo:", model_path)

        def init_env(i_env):
            def thunk():
                return Strong_Kick_Gym(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)
        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([init_env(n_envs)])

        try:
            if "model_file" in args:
                model = PPO.load(args["model_file"], env=env, device="cpu")
            else:
                model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=64, learning_rate=3e-4, device="cpu")
            
            self.learn_model(model, total_steps, model_path, eval_env=eval_env, eval_freq=20480, save_freq=204800, backup_env_file=__file__)
        except KeyboardInterrupt:
            sleep(1)
            print("\nCtrl+C pressionado, abortando...")
            servers.kill()
            return
        
        env.close()
        eval_env.close()
        servers.kill()

    def test(self, args):
        server = Server(self.server_p - 1, self.monitor_p, 1)
        env = Strong_Kick_Gym(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)
        try:
            self.export_model(args["model_file"], args["model_file"] + ".pkl", False)
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()
        env.close()
        server.kill()