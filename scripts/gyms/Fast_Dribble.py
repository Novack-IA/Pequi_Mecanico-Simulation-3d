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
        # self.prev_joint_vel is no longer needed

    def calculate_reward(self, env, action: np.ndarray, prev_action: np.ndarray):
        """Calcula a recompensa total como uma soma ponderada de vários componentes."""
        if not hasattr(env.player, 'world') or not hasattr(env.player.world, 'robot'):
            return 0.0

        r = env.player.world.robot
        w = env.player.world

        stability_reward = self._calculate_stability_reward(r)
        goal_directed_reward = self._calculate_goal_directed_reward(r, w)
        effort_reward = self._calculate_effort_reward(r, action, prev_action)

        total_reward = (self.weights.get('w_stability', 1.0) * stability_reward +
                        self.weights.get('w_goal', 1.5) * goal_directed_reward +
                        self.weights.get('w_effort', 1.0) * effort_reward)

        return total_reward

    def _calculate_stability_reward(self, robot):
        """
        Calcula a recompensa pela estabilidade do tronco.
        Incentiva o robô a permanecer ereto.
        """
        try:
            orientation = robot.imu_torso_orientation
            if isinstance(orientation, (list, tuple)) and len(orientation) >= 2:
                torso_roll, torso_pitch = orientation[0], orientation[1]
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0 

        stability_penalty = np.exp(-self.weights['stability_factor'] * (torso_roll**2 + torso_pitch**2))
        return stability_penalty * self.weights['stability']

    def _calculate_goal_directed_reward(self, robot, world):
        """
        Recompensa o robô por se mover em direção à bola e ao gol adversário.
        """
        ball_dist = np.linalg.norm(world.ball_rel_torso_cart_pos)
        r_ball_prox = np.exp(-(ball_dist**2) / self.weights['sigma_ball']**2)

        goal_pos = np.array([15.0, 0.0])
        ball_to_goal_dist = np.linalg.norm(world.ball_abs_pos[:2] - goal_pos)
        r_ball_goal = 0.0
        if self.prev_ball_to_goal_dist is not None:
            progress = self.prev_ball_to_goal_dist - ball_to_goal_dist
            r_ball_goal = progress * 50
        self.prev_ball_to_goal_dist = ball_to_goal_dist

        target_pos = world.ball_abs_pos
        agent_pos = robot.loc_head_position
        agent_vel = robot.get_head_abs_vel(2)
        target_dir = (target_pos - agent_pos) / (np.linalg.norm(target_pos - agent_pos) + 1e-6)
        r_vel_align = np.dot(agent_vel[:2], target_dir[:2])

        return (self.weights.get('w_ball_prox', 0.3) * r_ball_prox +
                self.weights.get('w_ball_goal', 0.6) * r_ball_goal +
                self.weights.get('w_vel_align', 0.1) * r_vel_align)

    def _calculate_effort_reward(self, robot, action, prev_action):
        """
        Penaliza o uso excessivo de torque e movimentos bruscos, promovendo eficiência.
        """
        # --- UPDATE: JOINT ACCELERATION REMOVED ---
        # Since joint velocities are not available from the parser, the joint
        # acceleration penalty (r_accel) has been removed. The remaining penalties
        # for torque and action rate still effectively encourage smooth movements.

        # Penalidade de Torque: penaliza o esforço físico (magnitude da ação).
        r_torque = -np.linalg.norm(action)**2

        # Penalidade de Taxa de Ação: penaliza mudanças bruscas nos comandos.
        r_action_rate = -np.linalg.norm(action - prev_action)**2

        return (self.weights.get('w_torque', 1e-5) * r_torque +
                self.weights.get('w_action_rate', 0.01) * r_action_rate)

    def reset(self):
        """Reseta os estados internos do calculador de recompensa."""
        self.prev_ball_to_goal_dist = None
        # self.prev_joint_vel is no longer needed

# =======================================================================================
# AMBIENTE GYM OTIMIZADO
# =======================================================================================
class Fast_Dribble_Gym(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw, curriculum_stage=4, reward_weights=None):
        if r_type != 4:
            print("AVISO: Este ambiente de treino é otimizado para o Robô Tipo 4.")

        self.robot_type = r_type
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0

        self.action_smoothing_factor = 0.2
        self.smoothed_action = None

        obs_size = 50
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        act_size = 22
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_size,), dtype=np.float32)

        self.curriculum_stage = curriculum_stage
        self.reward_weights = self._get_reward_weights(reward_weights)
        self.reward_calculator = RewardCalculator(self.reward_weights)

        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _get_reward_weights(self, custom_weights):
        """Define os pesos da recompensa para o estágio atual do currículo."""
        if custom_weights:
            return custom_weights
        
        # Default weights for the final curriculum stage
        return {
            'w_stability': 1.0, 'w_goal': 1.5, 'w_effort': 1.0,
            'stability_factor': 1.2, 'stability': 0.25,
            'w_ball_prox': 0.3, 'w_ball_goal': 0.6, 'w_vel_align': 0.1,
            'w_torque': 1e-5, 'w_accel': 2.5e-7, 'w_action_rate': 0.01,
            'sigma_ball': 0.5
        }

    def observe(self):
        """Constrói o vetor de observação de forma robusta."""
        obs = np.zeros(50, dtype=np.float32)
        
        # Return zero observation if the world is not ready
        if not hasattr(self.player, 'world'):
            return obs

        r = self.player.world.robot
        w = self.player.world

        # --- Safe data extraction ---
        try:
            # Robot state
            obs[0:3] = r.get_head_abs_vel(2) / 5.0
            
            # Safely handle orientation
            orientation = r.imu_torso_orientation
            if isinstance(orientation, (list, tuple)) and len(orientation) == 3:
                obs[3:6] = np.array(orientation) / 180.0
            
            obs[6:9] = np.array(r.gyro) / 200.0
            obs[9:12] = np.array(r.acc) / 10.0
            
            # Safely handle foot pressure sensors
            obs[12:15] = np.array(r.frp.get('lf', (0,0,0,0,0,0))[3:]) / 100.0
            obs[15:18] = np.array(r.frp.get('rf', (0,0,0,0,0,0))[3:]) / 100.0
            
            # Joint positions
            obs[18:40] = np.array(r.joints_position[2:24]) / 180.0
            
            # Ball state
            ball_rel_pos = np.array(w.ball_rel_torso_cart_pos)
            obs[40:43] = ball_rel_pos
            obs[43] = np.linalg.norm(ball_rel_pos) # Corrected index
            ideal_ball_pos = np.array([0.18, 0.0])
            obs[44] = np.linalg.norm(ball_rel_pos[:2] - ideal_ball_pos) # Corrected index

        except Exception as e:
            print(f"ERRO ao construir observação: {e}. Retornando observação nula.")
            return np.zeros(50, dtype=np.float32)
            
        return obs

    def sync(self):
        self.player.scom.commit_and_send(self.player.world.robot.get_command())
        self.player.scom.receive()

    def reset(self):
        self.step_counter = 0
        r = self.player.world.robot

        start_x = np.random.uniform(-5, 5)
        start_y = np.random.uniform(-3, 3)
        ball_offset_x = np.random.uniform(0.2, 0.3)
        ball_offset_y = np.random.uniform(-0.1, 0.1)

        self.player.scom.unofficial_beam((start_x, start_y, r.beam_height), 0)
        self.player.scom.unofficial_move_ball((start_x + ball_offset_x, start_y + ball_offset_y, 0.042))
        
        for _ in range(10):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()

        self.reward_calculator.reset()
        self.smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self.observe()

    def step(self, action):
        # Apply action smoothing
        if self.smoothed_action is None:
            self.smoothed_action = action
        else:
            self.smoothed_action = (self.action_smoothing_factor * action +
                                    (1 - self.action_smoothing_factor) * self.smoothed_action)

        r = self.player.world.robot
        r.set_joints_target_position_direct(slice(2, 24), self.smoothed_action * 25, harmonize=False)
        self.sync()
        self.step_counter += 1

        reward = self.reward_calculator.calculate_reward(self, self.smoothed_action, self.prev_action)
        self.prev_action = np.copy(self.smoothed_action)

        terminal = False
        info = {}
        
        if r.loc_head_z <= 0.4:
            reward -= 10.0
            terminal = True
        elif np.linalg.norm(self.player.world.ball_rel_torso_cart_pos) > 2.0:
            reward -= 5.0
            terminal = False
        elif self.step_counter > 1000:
            terminal = True
        
        return self.observe(), reward, terminal, info

    def render(self, mode='human', close=False):
        pass

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
        n_envs = max(1, os.cpu_count() - 2)
        total_steps = 25_000_000
        n_steps_per_env = 4096
        batch_size = 2048
        
        folder_name = f'Advanced_Dribble_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'
        os.makedirs(model_path, exist_ok=True)

        print(f"Iniciando treinamento com {n_envs} ambientes em paralelo.")
        print("Caminho do Modelo:", model_path)

        servers = None
        env = None
        eval_env = None
        
        try:
            # Start N servers for training + 1 for evaluation
            servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)

            # --- FIX: CORRECT PORT LOGIC ---
            # This function now correctly calculates the monitor port for each environment
            # by incrementing it sequentially, matching how the servers were started.
            def init_env(i_env):
                def thunk():
                    env = Fast_Dribble_Gym(self.ip, 
                                           self.server_p + i_env, 
                                           self.monitor_p_1000 + i_env, # Correct sequential port
                                           self.robot_type, 
                                           enable_draw=False, 
                                           curriculum_stage=4)
                    return env
                return thunk

            # Create the vectorized environments for training
            env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
            
            # Create a separate environment for evaluation
            eval_env = SubprocVecEnv([init_env(n_envs)]) # Uses the next available port
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)

            device = "cpu"
            print(f"Dispositivo de treinamento selecionado: {device.upper()}")

            model = None
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
            
            checkpoint_callback = CheckpointCallback(save_freq=max(20480 // n_envs, 1),
                                                     save_path=model_path,
                                                     name_prefix='dribble_agent')

            print("Iniciando o aprendizado do modelo...")
            model.learn(total_timesteps=total_steps, callback=checkpoint_callback)

        except (KeyboardInterrupt, Exception) as e:
            print(f"\nOcorreu uma exceção: {e}. Salvando modelo e abortando...")
            if 'model' in locals() and model is not None:
                model.save(f"{model_path}/dribble_agent_aborted.zip")
            if env:
                env.save(f"{model_path}/vec_normalize_aborted.pkl")

        finally:
            print("Fechando ambientes e servidores...")
            if env: env.close()
            if eval_env: eval_env.close()
            if servers: servers.kill()
            print("Treinamento concluído.")

    def test(self, args):
        server = None
        env = None
        try:
            server = Server(self.server_p - 1, self.monitor_p, 1)
            
            vec_normalize_path = os.path.join(args["folder_dir"], "vec_normalize.pkl")
            model_file_path = args["model_file"]

            if not os.path.exists(vec_normalize_path) or not os.path.exists(model_file_path):
                print(f"ERRO: Não foi possível encontrar o modelo '{model_file_path}' ou o normalizador '{vec_normalize_path}'")
                return

            def init_test_env():
                return Fast_Dribble_Gym(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
            
            env = SubprocVecEnv([init_test_env])
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False

            model = PPO.load(model_file_path, env=env)

            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
            
        except KeyboardInterrupt:
            print("\nTeste interrompido pelo usuário.")
        except Exception as e:
            print(f"Ocorreu um erro durante o teste: {e}")
        finally:
            if env: env.close()
            if server: server.kill()
            print("Teste finalizado.")