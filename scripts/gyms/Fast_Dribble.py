# -*- coding: utf-8 -*-

from agent.Base_AgentCEIA import Base_Agent as Agent
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os
import gym
import numpy as np
from math_ops.Math_Ops import Math_Ops as M

'''
Objetivo:
Treinar o Robô Tipo 4 para um drible rápido, estável e eficiente em direção ao gol.
Esta versão implementa uma arquitetura de aprendizado baseada em pesquisa de ponta,
focando em três pilares:
1.  **Função de Recompensa Sofisticada:** Uma política de recompensa multi-objetivo que
    equilibra estabilidade, direcionamento ao gol, e eficiência de movimento.
2.  **Heurísticas de Aprendizagem Robustas:** Incorpora suavização de ações e
    randomização de domínio para criar uma política mais generalizável e suave.
3.  **Pipeline de Treinamento de Alta Performance:** Otimizado para máxima coleta de
    amostras através de paralelização massiva, a estratégia mais eficaz para
    acelerar o treinamento em simuladores baseados em CPU como o simspark.
'''

# =======================================================================================
# MÓDULO DE CÁLCULO DE RECOMPENSA
# =======================================================================================
class RewardCalculator:
    """
    Encapsula a lógica de cálculo da recompensa multi-objetivo.
    Isso torna o código mais modular e fácil de manter, permitindo que a função de
    recompensa seja ajustada independentemente do ambiente do ginásio.
    """
    def __init__(self, weights: dict):
        self.weights = weights
        self.prev_ball_to_goal_dist = None
        self.prev_joint_vel = None

    def calculate_reward(self, env, action: np.ndarray, prev_action: np.ndarray):
        """Calcula a recompensa total como uma soma ponderada de vários componentes."""
        r = env.player.world.robot
        w = env.player.world

        # --- 1. Componente de Estabilidade (Manter-se em pé e equilibrado) ---
        stability_reward = self._calculate_stability_reward(r)

        # --- 2. Componente Direcionado a Objetivos (Mover-se com propósito) ---
        goal_directed_reward = self._calculate_goal_directed_reward(r, w)

        # --- 3. Componente de Eficiência (Movimento suave e econômico) ---
        effort_reward = self._calculate_effort_reward(r, action, prev_action)

        # Combina os componentes com seus respectivos pesos
        total_reward = (self.weights.get('w_stability', 1.0) * stability_reward +
                        self.weights.get('w_goal', 1.0) * goal_directed_reward +
                        self.weights.get('w_effort', 1.0) * effort_reward)

        return total_reward

    def _calculate_stability_reward(self, robot):
        """
        Penaliza o robô por instabilidade. A base para qualquer habilidade. [1, 2, 3]
        """
        # Penalidade por orientação do tronco: incentiva a manter o tronco ereto. 
        torso_roll, torso_pitch = robot.imu_torso_orientation, robot.imu_torso_orientation[4]
        r_orient = np.exp(-75.0 * (torso_roll**2 + torso_pitch**2))

        # Penalidade por velocidade vertical: desencoraja saltos ou movimentos bruscos. [5]
        torso_vz = robot.get_head_abs_vel(2)[6]
        r_vert_vel = -torso_vz**2

        # Penalidade por altura do corpo: incentiva uma postura atlética consistente. [5]
        desired_height = 0.65 # Altura alvo para o tronco
        r_height = -(robot.loc_head_z - desired_height)**2

        return (self.weights.get('w_orient', 0.4) * r_orient +
                self.weights.get('w_vert_vel', 0.2) * r_vert_vel +
                self.weights.get('w_height', 0.4) * r_height)

    def _calculate_goal_directed_reward(self, robot, world):
        """
        Recompensa o robô por se mover em direção à bola e ao gol adversário. [7, 8]
        """
        # Proximidade Agente-Bola: recompensa por se aproximar da bola (Gaussiana). [9]
        ball_dist = np.linalg.norm(world.ball_rel_torso_cart_pos)
        sigma_ball = 0.5 # Controla a "zona de interesse"
        r_ball_prox = np.exp(-(ball_dist**2) / sigma_ball**2)

        # Progresso Bola-Gol: recompensa por mover a bola na direção certa.
        goal_pos = np.array([15.0, 0.0]) # Posição do gol adversário
        ball_to_goal_dist = np.linalg.norm(world.ball_abs_pos[:2] - goal_pos)
        r_ball_goal = 0.0
        if self.prev_ball_to_goal_dist is not None:
            r_ball_goal = -(ball_to_goal_dist - self.prev_ball_to_goal_dist) * 50 # Forte incentivo
        self.prev_ball_to_goal_dist = ball_to_goal_dist

        # Alinhamento da Velocidade: recompensa por se mover eficientemente para o alvo. [10, 11, 12]
        target_pos = world.ball_abs_pos
        agent_pos = robot.loc_head_position
        agent_vel = robot.get_head_abs_vel(2)
        target_dir = (target_pos - agent_pos) / (np.linalg.norm(target_pos - agent_pos) + 1e-6)
        r_vel_align = np.dot(agent_vel[:2], target_dir[:2]) # Apenas no plano XY

        return (self.weights.get('w_ball_prox', 0.3) * r_ball_prox +
                self.weights.get('w_ball_goal', 0.5) * r_ball_goal +
                self.weights.get('w_vel_align', 0.2) * r_vel_align)

    def _calculate_effort_reward(self, robot, action, prev_action):
        """
        Penaliza o uso excessivo de torque e movimentos bruscos, promovendo eficiência. [5, 13, 14]
        """
        # Penalidade de Torque: penaliza o esforço físico dos motores. [15, 16, 17]
        # Nota: SimSpark não expõe torques diretamente, usamos a magnitude da ação como proxy.
        r_torque = -np.linalg.norm(action)**2

        # Penalidade de Aceleração das Juntas: promove suavidade.
        joint_vel = robot.joints_velocity[2:24]
        r_accel = 0.0
        if self.prev_joint_vel is not None:
            joint_accel = joint_vel - self.prev_joint_vel
            r_accel = -np.linalg.norm(joint_accel)**2
        self.prev_joint_vel = joint_vel

        # Penalidade de Taxa de Ação: regulariza a política para comandos mais suaves. [5]
        r_action_rate = -np.linalg.norm(action - prev_action)**2

        return (self.weights.get('w_torque', 1e-5) * r_torque +
                self.weights.get('w_accel', 2.5e-7) * r_accel +
                self.weights.get('w_action_rate', 0.01) * r_action_rate)

    def reset(self):
        """Reseta os estados internos do calculador de recompensa."""
        self.prev_ball_to_goal_dist = None
        self.prev_joint_vel = None

# =======================================================================================
# AMBIENTE GYM OTIMIZADO
# =======================================================================================
class Fast_Dribble_Gym(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw, curriculum_stage=4, reward_weights=None):
        if r_type!= 4:
            print("AVISO: Este ambiente de treino é otimizado para o Robô Tipo 4.")

        self.robot_type = r_type
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0

        # --- Heurística: Suavização de Ação (Action Smoothing) ---
        # Aplica um filtro de média móvel exponencial para suavizar os comandos da política,
        # resultando em movimentos mais estáveis e naturais. 
        self.action_smoothing_factor = 0.2
        self.smoothed_action = None

        # --- Espaço de Observação ---
        # Focado em informações dinâmicas de alta frequência para decisões rápidas
        obs_size = 50
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # --- Espaço de Ação ---
        # Controle direto sobre as 22 articulações do corpo (excluindo cabeça)
        act_size = 22
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_size,), dtype=np.float32)

        # --- Configuração da Recompensa e Currículo ---
        # A lógica de recompensa é modularizada para clareza e para facilitar o
        # aprendizado por currículo, onde os pesos podem mudar a cada estágio. [18, 19, 20]
        self.curriculum_stage = curriculum_stage
        self.reward_weights = self._get_reward_weights(reward_weights)
        self.reward_calculator = RewardCalculator(self.reward_weights)

        # Armazena o estado anterior para cálculos de recompensa (ex: aceleração)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _get_reward_weights(self, custom_weights):
        """Define os pesos da recompensa para o estágio atual do currículo."""
        if custom_weights:
            return custom_weights

        # Pesos para o estágio final: Drible Completo
        if self.curriculum_stage == 4:
            return {
                'w_stability': 1.0, 'w_goal': 1.5, 'w_effort': 1.0,
                'w_orient': 0.4, 'w_vert_vel': 0.2, 'w_height': 0.4,
                'w_ball_prox': 0.3, 'w_ball_goal': 0.6, 'w_vel_align': 0.1,
                'w_torque': 1e-5, 'w_accel': 2.5e-7, 'w_action_rate': 0.01,
                'desired_height': 0.65, 'sigma_ball': 0.5
            }
        # Outros estágios (Estabilidade, Locomoção, etc.) seriam definidos aqui
        # com diferentes pesos para focar em habilidades específicas.
        else:
            # Padrão para o estágio final se não especificado
            return self.get_reward_weights(None)

    def observe(self):
        r = self.player.world.robot
        w = self.player.world
        obs = np.zeros(50, dtype=np.float32)

        # Normalização é crucial para o bom desempenho de redes neurais.
        # Os valores são escalados para uma faixa aproximadamente entre [-1, 1].
        obs[0:3]   = r.get_head_abs_vel(2) / 5.0      # Velocidade do robô
        obs[3:6]   = r.imu_torso_orientation / 180.0  # Orientação do tronco (roll, pitch, yaw)
        obs[6:9]   = r.gyro / 200.0                   # Velocidade angular
        obs[9:12]  = r.acc / 10.0                     # Aceleração linear
        obs[12:15] = r.frp.get('lf', (0,0,0,0,0,0))[3:] / 100.0 # Força no pé esquerdo
        obs[15:18] = r.frp.get('rf', (0,0,0,0,0,0))[3:] / 100.0 # Força no pé direito
        obs[18:40] = r.joints_position[2:24] / 180.0  # Posição das juntas
        obs[40:43] = w.ball_rel_torso_cart_pos        # Posição relativa da bola
        obs[21]    = np.linalg.norm(w.ball_rel_torso_cart_pos) # Distância da bola
        ideal_ball_pos = np.array([0.18, 0.0])
        obs[22] = np.linalg.norm(w.ball_rel_torso_cart_pos[:2] - ideal_ball_pos) # Distância da posição ideal

        return obs

    def sync(self):
        self.player.scom.commit_and_send(self.player.world.robot.get_command())
        self.player.scom.receive()

    def reset(self):
        self.step_counter = 0
        r = self.player.world.robot

        # --- Heurística: Randomização de Domínio (Domain Randomization) ---
        # Introduz pequenas variações no início de cada episódio para forçar a política
        # a aprender uma solução mais robusta e generalizável. [19, 20, 23]
        start_x = np.random.uniform(-5, 5)
        start_y = np.random.uniform(-3, 3)
        ball_offset_x = np.random.uniform(0.2, 0.3)
        ball_offset_y = np.random.uniform(-0.1, 0.1)

        self.player.scom.unofficial_beam((start_x, start_y, r.beam_height), 0)
        self.player.scom.unofficial_move_ball((start_x + ball_offset_x, start_y + ball_offset_y, 0.042))
        
        # Aplica uma pequena força aleatória para simular perturbações
        push_force_x = np.random.uniform(-10, 10)
        push_force_y = np.random.uniform(-10, 10)
        self.player.scom.apply_force_to_agent(1, (push_force_x, push_force_y, 0))

        for _ in range(10):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()

        self.reward_calculator.reset()
        self.smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self.observe()

    def step(self, action):
        # Aplica suavização de ação
        if self.smoothed_action is None:
            self.smoothed_action = action
        else:
            self.smoothed_action = (self.action_smoothing_factor * action +
                                    (1 - self.action_smoothing_factor) * self.smoothed_action)

        # Ação direta nas articulações (agora escalada de [-1, 1] para um range de graus)
        r = self.player.world.robot
        r.set_joints_target_position_direct(slice(2, 24), self.smoothed_action * 25, harmonize=False)
        self.sync()
        self.step_counter += 1

        # Calcula a recompensa usando o módulo dedicado
        reward = self.reward_calculator.calculate_reward(self, self.smoothed_action, self.prev_action)
        self.prev_action = self.smoothed_action

        # --- Condições de Término (Eventos) ---
        # Recompensas/penalidades esparsas para eventos chave.
        terminal = False
        if r.loc_head_z <= 0.4: # Penalidade por cair
            reward -= 10.0
            terminal = True
        elif np.linalg.norm(self.player.world.ball_rel_torso_cart_pos) > 2.0: # Penalidade por perder a bola
            reward -= 5.0
            terminal = False # Não termina, para aprender a recuperar
        elif self.step_counter > 1000: # Fim do episódio por tempo
            terminal = True
        
        # Bônus por tocar na bola (se aplicável no currículo)
        # Bônus por marcar gol (se aplicável no currículo)

        return self.observe(), reward, terminal, {}

    def render(self, mode='human', close=False): return

    def close(self):
        Draw.clear_all()
        self.player.terminate()

# =======================================================================================
# CLASSE DE TREINAMENTO OTIMIZADA
# =======================================================================================
class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):
        # --- Otimização de Pipeline: Paralelização Massiva ---
        # Acelerar o treinamento em simuladores como o simspark não vem do uso da GPU,
        # mas sim do aumento da taxa de coleta de experiências. Usar múltiplos
        # ambientes em paralelo em núcleos de CPU é a estratégia mais eficaz. [4, 24]
        # O gargalo é a transferência de dados CPU-GPU a cada passo, que é mais lenta
        # que a própria simulação. [21, 22, 25]
        n_envs = max(1, os.cpu_count() - 2) # Use quase todos os cores disponíveis
        total_steps = 25_000_000 # Total de passos de treinamento
        
        # Hiperparâmetros do PPO ajustados para treinamento em paralelo
        n_steps_per_env = 4096 # Mais passos por coleta antes da atualização
        batch_size = 2048      # Batch size maior para atualizações mais estáveis
        
        folder_name = f'Advanced_Dribble_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'
        os.makedirs(model_path, exist_ok=True)

        print(f"Iniciando treinamento com {n_envs} ambientes em paralelo.")
        print("Caminho do Modelo:", model_path)

        def init_env(i_env):
            def thunk():
                # Cada ambiente pode ter configurações diferentes para o currículo
                env = Fast_Dribble_Gym(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False, curriculum_stage=4)
                return env
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)
        
        # Cria os ambientes vetorizados
        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        # Normaliza as observações, uma prática recomendada para RL
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        eval_env = SubprocVecEnv([init_env(n_envs)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)

        # --- Seleção Inteligente de Dispositivo (CPU vs GPU) ---
        # Para simspark, o treinamento na CPU é quase sempre mais rápido devido ao
        # gargalo de transferência de dados. A GPU só se torna vantajosa com um
        # número massivo de ambientes (>128) e redes muito grandes (ex: com visão).
        # Comece com 'cpu' e só mude para 'cuda' se o gargalo for a atualização do modelo. [26]
        device = "cpu"
        print(f"Dispositivo de treinamento selecionado: {device.upper()}")

        try:
            if "model_file" in args:
                print(f"Carregando modelo de: {args['model_file']}")
                model = PPO.load(args["model_file"], env=env, device=device)
            else:
                print("Criando um novo modelo PPO.")
                model = PPO("MlpPolicy",
                            env,
                            verbose=1,
                            n_steps=n_steps_per_env,
                            batch_size=batch_size,
                            learning_rate=3e-4,
                            gamma=0.99,
                            ent_coef=0.0,
                            n_epochs=10,
                            device=device,
                            tensorboard_log=f'./scripts/gyms/tensorboard/{folder_name}/')
            
            # Callback para salvar o modelo e o estado do VecNormalize
            checkpoint_callback = CheckpointCallback(save_freq=max(20480 // n_envs, 1),
                                                     save_path=model_path,
                                                     name_prefix='dribble_agent')

            print("Iniciando o aprendizado do modelo...")
            model.learn(total_timesteps=total_steps, callback=checkpoint_callback)

        except KeyboardInterrupt:
            sleep(1)
            print("\nCtrl+C pressionado, salvando modelo e abortando...")
            model.save(f"{model_path}/dribble_agent_interrupted.zip")
            env.save(f"{model_path}/vec_normalize_interrupted.pkl")

        finally:
            print("Fechando ambientes e servidores...")
            env.close()
            eval_env.close()
            servers.kill()
            print("Treinamento concluído.")

    def test(self, args):
        server = Server(self.server_p - 1, self.monitor_p, 1)
        
        # Carrega o VecNormalize salvo durante o treinamento
        vec_normalize_path = os.path.join(args["folder_dir"], "vec_normalize.pkl")
        if not os.path.exists(vec_normalize_path):
             # Fallback para o último salvo se o principal não existir
             vec_normalize_path = os.path.join(args["folder_dir"], "rl_model_vecnormalize_..._steps.pkl") # Adapte o nome
        
        # Cria um ambiente de teste e o envolve com o estado de normalização carregado
        env = Fast_Dribble_Gym(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        env = SubprocVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False # Desativa a atualização das estatísticas de normalização
        env.norm_reward = False

        model = PPO.load(args["model_file"], env=env)
        try:
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()
        finally:
            env.close()
            server.kill()