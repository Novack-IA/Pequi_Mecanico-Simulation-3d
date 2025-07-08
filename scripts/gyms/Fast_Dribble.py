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
Treinar o Robô Tipo 4 para uma corrida/drible rápido e controlado em direção ao gol,
preparando para uma transição suave para o chute.
----------
- class Fast_Dribble_Gym: implementa um ginásio personalizado do OpenAI.
- class Train: implementa os algoritmos para treinar um novo modelo.
'''

class Fast_Dribble_Gym(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
        if r_type != 4:
            print("AVISO: Este ambiente de treino é otimizado para o Robô Tipo 4.")

        self.robot_type = r_type
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0

        # --- Espaço de Observação ---
        # Focado em informações dinâmicas de alta frequência
        obs_size = 50
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # --- Espaço de Ação ---
        # Controle total sobre as 24 articulações (incluindo os dedos para o Tipo 4)
        MAX = np.finfo(np.float32).max
        act_size = 22 # Excluindo a cabeça
        self.action_space = gym.spaces.Box(low=-MAX, high=MAX, shape=(act_size,), dtype=np.float32)

    def observe(self):
        r = self.player.world.robot
        w = self.player.world
        
        obs = np.zeros(50, dtype=np.float32)

        # Estado do Robô (15)
        obs[0:3]   = r.get_head_abs_vel(2) * 2.0  # Velocidade do robô (com maior peso)
        obs[3:6]   = r.imu_torso_orientation / 180.0
        obs[6:9]   = r.gyro / 200.0
        obs[9:12]  = r.acc / 10.0
        obs[12:15] = r.frp.get('lf', (0,0,0,0,0,0))[3:] # Força no pé esquerdo
        obs[15:18] = r.frp.get('rf', (0,0,0,0,0,0))[3:] # Força no pé direito

        # Estado das Articulações (24)
        obs[18:40] = r.joints_position[2:24] / 180.0

        # Estado da Bola (5)
        ball_rel_pos = w.ball_rel_torso_cart_pos
        obs[40:43] = ball_rel_pos
        obs[43]    = np.linalg.norm(ball_rel_pos)
        
        # Posição ideal da bola para o drible (um pouco à frente e centrada)
        ideal_ball_pos = np.array([0.18, 0.0])
        obs[44] = np.linalg.norm(ball_rel_pos[:2] - ideal_ball_pos)

        return obs

    def sync(self):
        self.player.scom.commit_and_send(self.player.world.robot.get_command())
        self.player.scom.receive()

    def reset(self):
        self.step_counter = 0
        r = self.player.world.robot

        # Posição inicial: robô um pouco atrás da bola no meio do campo
        start_x = np.random.uniform(-5, 5)
        start_y = np.random.uniform(-3, 3)
        self.player.scom.unofficial_beam((start_x, start_y, r.beam_height), 0)
        self.player.scom.unofficial_move_ball((start_x + 0.2, start_y, 0.042))
        
        for _ in range(10):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()

        self.last_robot_x = r.loc_head_position[0]
        self.last_ball_x = self.player.world.ball_abs_pos[0]
        return self.observe()

    def step(self, action):
        r = self.player.world.robot
        w = self.player.world
        
        # Ação direta nas articulações
        r.set_joints_target_position_direct(slice(2, 24), action * 25, harmonize=False)
        self.sync()
        self.step_counter += 1
        
        # --- Função de Recompensa Otimizada ---
        
        # 1. Recompensa por avanço (velocidade)
        robot_advancement = r.loc_head_position[0] - self.last_robot_x
        ball_advancement = w.ball_abs_pos[0] - self.last_ball_x
        self.last_robot_x = r.loc_head_position[0]
        self.last_ball_x = w.ball_abs_pos[0]
        
        reward = robot_advancement * 15.0 # Forte incentivo para correr para frente
        
        # 2. Recompensa por controle da bola
        ball_rel_pos = w.ball_rel_torso_cart_pos
        ball_dist = np.linalg.norm(ball_rel_pos[:2])
        
        # Incentiva a manter a bola perto
        reward += (0.4 - ball_dist) * 5.0
        
        # Incentiva a bola a avançar junto com o robô
        if ball_advancement > 0:
            reward += ball_advancement * 15.0
            
        # 3. Penalidades
        if r.loc_head_z < 0.3: # Penalidade por cair
            reward = -20 
            terminal = True
        elif ball_dist > 0.6: # Penalidade por perder a bola
            reward = -10
            terminal = False
        elif self.step_counter > 500: # Fim do episódio por tempo
            terminal = True
        else:
            terminal = False

        return self.observe(), reward, terminal, {}
    
    # ... outros métodos ...
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
        total_steps = 20_000_000
        folder_name = f'Fast_Dribble_Gym_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Caminho do Modelo:", model_path)

        def init_env(i_env):
            def thunk():
                return Fast_Dribble_Gym(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
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
        env = Fast_Dribble_Gym(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)
        try:
            self.export_model(args["model_file"], args["model_file"] + ".pkl", False)
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()
        env.close()
        server.kill()