# Salve este arquivo como: scripts/gyms/Strong_Kick.py

from agent.Base_AgentCEIA import Base_Agent as Agent
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os, gym
import numpy as np

'''
Objetivo Revisado:
Treinar o Robô Tipo 4 para executar um chute FORTE e PRECISO, aprendendo
o movimento completo de aproximação e chute de forma estável.
----------
- class Strong_Kick_Gym: implementa um ginásio personalizado do OpenAI.
- class Train: implementa os algoritmos para treinar um novo modelo.
'''

class Strong_Kick_Gym(gym.Env):
	def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
		if r_type != 4:
			print("AVISO: Este ambiente de treino é otimizado para o Robô Tipo 4.")

		self.robot_type = r_type
		self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
		self.step_counter = 0

		# --- Espaço de Observação (42) ---
		# Mantido, pois é um bom conjunto de informações.
		obs_size = 42
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

		# --- Espaço de Ação (12) ---
		# Controlará as 12 articulações das pernas e pés (llj1 a rlj6)
		act_size = 12
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(act_size,), dtype=np.float32) # Ações normalizadas entre -1 e 1

		self.kicking_foot = 'l' # Define o pé de chute ('l' ou 'r')

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

		# Estado da Bola em relação ao Torso (3)
		ball_rel_pos = w.ball_rel_torso_cart_pos
		obs[30:33] = ball_rel_pos
		
		# --- CORREÇÃO APLICADA AQUI ---
		# A posição do pé é obtida do sensor de força (frp), que inclui o ponto de origem.
		foot_key = self.kicking_foot + 'f'  # Constrói a chave 'lf' ou 'rf'
		foot_data = r.frp.get(foot_key, (0, 0, 0, 0, 0, 0))
		foot_pos = foot_data[:3]  # Os 3 primeiros valores são a posição (px, py, pz)
		
		ball_rel_foot_pos = ball_rel_pos - np.array(foot_pos)
		obs[33] = np.linalg.norm(ball_rel_foot_pos) # Distância
		obs[34] = np.arctan2(ball_rel_foot_pos[1], ball_rel_foot_pos[0]) # Ângulo
		# --- FIM DA CORREÇÃO ---

		# Posições do pé de apoio e do pé de chute (6)
		# Usando a mesma lógica para ambos os pés
		obs[35:38] = r.frp.get('lf', (0,0,0,0,0,0))[:3]
		obs[38:41] = r.frp.get('rf', (0,0,0,0,0,0))[:3]

		return obs

	def sync(self):
		self.player.scom.commit_and_send(self.player.world.robot.get_command())
		self.player.scom.receive()

	def reset(self):
		self.step_counter = 0
		r = self.player.world.robot
		w = self.player.world

		# --- Reset Robusto (inspirado no Fall.py) para garantir estabilidade ---

		# 1. Estabiliza o robô flutuando no ar primeiro, para zerar qualquer momento residual.
		for _ in range(25):
			self.player.scom.unofficial_beam((13.8, 0, 0.50), 180) # Posição alta
			self.player.behavior.execute("Zero")
			self.sync()

		# 2. Posiciona o robô no chão (na altura correta dos pés).
		self.player.scom.unofficial_beam((13.8, 0, r.beam_height), 180)
		
		# Adiciona a aleatoriedade na posição da bola.
		ball_x = 14.0 + np.random.uniform(-0.05, 0.05)
		ball_y = -0.09 + np.random.uniform(-0.03, 0.03)
		self.player.scom.unofficial_move_ball((ball_x, ball_y, 0.042))

		# 3. Estabiliza o robô no chão por alguns passos.
		for _ in range(10):
			self.player.behavior.execute("Zero_Bent_Knees") # Usar joelhos flexionados ajuda na estabilidade inicial
			self.sync()

		return self.observe()

	def step(self, action):
		r = self.player.world.robot
		w = self.player.world
		
		# --- Ação Relativa (Delta Action) para Movimentos Suaves ---
		# A IA não define a posição final, mas sim uma *mudança* a partir da posição atual.
		# Isso evita movimentos bruscos e desequilíbrios.
		current_positions = r.joints_position[2:14]
		# Uma escala menor para a ação resulta em movimentos mais finos e estáveis.
		new_target_positions = current_positions + action * 5.0 # Escala reduzida de 30 para 5
		r.set_joints_target_position_direct(slice(2, 14), new_target_positions, harmonize=False)

		self.sync()
		self.step_counter += 1
		
		obs = self.observe()
		
		# --- Função de Recompensa Modelada (Reward Shaping) ---
		# Feedback contínuo para guiar o agente de forma estável.

		# Recompensa por se aproximar da bola com o pé correto
		dist_foot_ball = obs[33]
		reward_proximity = (1.0 / (1.0 + dist_foot_ball**2)) * 0.1 # Recompensa pequena, mas constante

		# Recompensa pela velocidade da bola (a principal, após o chute)
		ball_vel = w.get_ball_abs_vel(1)
		reward_kick_speed = max(0, ball_vel[0]) # Apenas recompensa velocidade para frente

		# Penalidade por desvio (incentiva precisão)
		penalty_deviation = abs(ball_vel[1]) * 0.5

		# Combina as recompensas
		reward = reward_proximity + reward_kick_speed - penalty_deviation
		
		# --- Condições de Fim de Episódio com Recompensas Menos "Explosivas" ---
		terminal = False
		if r.loc_head_z < 0.3: # Penalidade por cair
			reward = -1.0 # Penalidade pequena, mas clara
			terminal = True
		elif w.ball_abs_pos[0] > 15.05 and abs(w.ball_abs_pos[1]) < 1.05: # Gol
			reward = 5.0 # Bônus, mas não tão grande a ponto de desestabilizar
			terminal = True
		elif self.step_counter > 200: # Timeout
			terminal = True

		return obs, reward, terminal, {}
	
	def render(self, mode='human', close=False): return
	def close(self):
		Draw.clear_all()
		self.player.terminate()

class Train(Train_Base):
	def __init__(self, script) -> None:
		super().__init__(script)

	def train(self, args):
		n_envs = min(12, os.cpu_count())
		# Aumentar o n_steps_per_env dá ao agente mais dados por atualização, estabilizando o treino
		n_steps_per_env = 2048 
		total_steps = 25_000_000
		folder_name = f'Strong_Kick_R{self.robot_type}_v2'
		model_path = f'./scripts/gyms/logs/{folder_name}/'

		print("Caminho do Modelo:", model_path)

		def init_env(i_env):
			def thunk():
				# A classe agora é Strong_Kick_Gym
				return Strong_Kick_Gym(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
			return thunk

		servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)
		env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
		eval_env = SubprocVecEnv([init_env(n_envs)])

		try:
			if "model_file" in args:
				model = PPO.load(args["model_file"], env=env, device="cpu")
			else:
				# Hiperparâmetros padrão do PPO que são geralmente mais estáveis
				model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, device="cpu")
			
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