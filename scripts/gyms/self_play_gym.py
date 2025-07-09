import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from scripts.commons.Script import Script
from agent.Base_AgentCEIA import Base_Agent as Agent
from math_ops.Math_Ops import Math_Ops as M
import os

class SelfPlayEnv(gym.Env):
    """
    Ambiente de Gym customizado para treinamento de self-play 11 contra 11.
    """
    def __init__(self, script, opponent_model_path=None):
        super(SelfPlayEnv, self).__init__()
        self.script = script
        self.a = self.script.args

        # Criando os 11 jogadores do time aprendiz (Home)
        self.script.batch_create(Agent, ((self.a.i, self.a.p, self.a.m, u + 1, self.a.r, "Home") for u in range(11)))
        self.home_players = self.script.players[-11:]

        # Criando os 11 jogadores do time adversário (Away)
        self.script.batch_create(Agent, ((self.a.i, self.a.p, self.a.m, u + 1, self.a.r, "Away") for u in range(11)))
        self.away_players = self.script.players[-11:]

        self.players = self.home_players + self.away_players
        self.p_num = len(self.players)

        # Carregar o modelo do adversário, se houver
        self.opponent_model = None
        if opponent_model_path and os.path.exists(opponent_model_path):
            print(f"Carregando modelo do adversário de: {opponent_model_path}")
            self.opponent_model = PPO.load(opponent_model_path)

        # Espaço de Ação e Observação (simplificado, precisa ser ajustado para suas necessidades)
        # Exemplo: Ação para cada jogador (chutar, mover, etc.)
        # Exemplo: Observação (posição da bola, posições dos jogadores, etc.)
        self.action_space = gym.spaces.MultiDiscrete([5] * 11)  # 5 ações possíveis para cada um dos 11 jogadores
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(22, 3), dtype=np.float32) # Posição (x,y,z) de todos os 22 jogadores

        self.reset()

    def step(self, action):
        # 1. Executar a ação para o time aprendiz
        for i, player in enumerate(self.home_players):
            # Lógica para traduzir o 'action[i]' em um comportamento do robô
            # Ex: se action[i] == 0, chutar. Se 1, mover para a bola, etc.
            # Aqui, para simplificar, vamos assumir que a ação é chutar na direção do gol
            goal_dir = M.vector_angle((15, 0) - player.world.robot.loc_head_position[:2])
            player.behavior.execute("Basic_Kick", goal_dir)

        # 2. Obter e executar a ação para o time adversário
        if self.opponent_model:
            obs = self._get_observation()
            away_action, _ = self.opponent_model.predict(obs, deterministic=True)
            for i, player in enumerate(self.away_players):
                 goal_dir = M.vector_angle((15, 0) - player.world.robot.loc_head_position[:2])
                 player.behavior.execute("Basic_Kick", goal_dir)

        # 3. Executar um passo na simulação para todos os jogadores
        self.script.batch_commit_and_send()
        self.script.batch_receive()

        # 4. Calcular a recompensa
        reward, done = self._calculate_reward()

        # 5. Obter a nova observação
        observation = self._get_observation()
        
        info = {}
        return observation, reward, done, info

    def reset(self):
        # Reinicia a posição dos jogadores e da bola
        self.script.batch_unofficial_beam((-3, 0, 0.5, 0) for i in range(self.p_num))
        # Reinicia o estado do jogo para PlayOn
        self.players[0].scom.unofficial_set_play_mode("PlayOn")
        return self._get_observation()

    def _get_observation(self):
        # Coleta e retorna a observação do estado atual do jogo
        obs = []
        for p in self.players:
            obs.append(p.world.robot.loc_head_position)
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self):
        # Lógica de recompensa
        reward = 0
        done = False
        
        w_home = self.home_players[0].world
        
        # Penalidade por gol sofrido
        if w_home.game.play_mode == w_home.M_GOAL_THEIR:
            reward -= 10
            done = True

        # Recompensa por gol marcado
        if w_home.game.play_mode == w_home.M_GOAL_OUR:
            reward += 10
            done = True

        # Recompensa por posse de bola (exemplo simples)
        if w_home.ball.last_touch_side == 'left': # 'left' é o nosso time (Home)
            reward += 0.1

        # Penalidade por falta cometida (exemplo)
        # É preciso uma lógica para detectar faltas, que pode ser complexa.
        # if self._detect_foul():
        #    reward -= 1

        return reward, done

    def render(self, mode='human'):
        # A visualização é feita pelo RoboViz, então não precisamos implementar aqui.
        pass

    def close(self):
        # Mata os processos do servidor se necessário
        os.system("pkill rcssserver3d")

class Train:
    def __init__(self, script):
        self.script = script
        self.a = script.args

    def train(self, total_timesteps=50000, opponent_model_path=None, save_path_prefix="gen"):
        env = SelfPlayEnv(self.script, opponent_model_path)
        
        # Cria um diretório para salvar o modelo desta geração
        save_path = f"./models/{save_path_prefix}"
        os.makedirs(save_path, exist_ok=True)
        
        model = PPO("MlpPolicy", env, verbose=1)
        
        # Callback para salvar o melhor modelo
        eval_callback = EvalCallback(env, best_model_save_path=save_path,
                                     log_path=save_path, eval_freq=500,
                                     deterministic=True, render=False)

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save(f"{save_path}/final_model.zip")
        print(f"Modelo final salvo em: {save_path}/final_model.zip")
        print(f"Melhor modelo salvo em: {save_path}/best_model.zip")