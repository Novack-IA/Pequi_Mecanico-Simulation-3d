import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from scripts.commons.Script import Script
from agent.Base_Agent import Base_Agent as Agent # Garanta que o nome 'Base_Agent' está correto
from math_ops.Math_Ops import Math_Ops as M
import os
import time

class SelfPlayEnv(gym.Env):
    """
    Ambiente de Gym customizado para treinamento de self-play 11 contra 11.
    """
    def __init__(self, script, opponent_model_path=None):
        super(SelfPlayEnv, self).__init__()
        self.script = script
        self.a = self.script.args

        # --- CORREÇÃO APLICADA AQUI ---

        # Criando os 11 jogadores do time aprendiz (Home)
        # Eles usarão as portas a partir de a.p (ex: 3100)
        print(f"Criando time 'Home' na porta {self.a.p}...")
        self.script.batch_create(Agent, ((self.a.i, self.a.p, self.a.m, u + 1, self.a.r, "Home") for u in range(11)))
        self.home_players = self.script.players[-11:]

        # Dando um pequeno tempo para o servidor processar
        time.sleep(1) 

        # Criando os 11 jogadores do time adversário (Away)
        # Eles usarão as portas a partir de a.p + 11 (ex: 3111)
        away_team_port = self.a.p + 11 
        print(f"Criando time 'Away' na porta {away_team_port}...")
        self.script.batch_create(Agent, ((self.a.i, away_team_port, self.a.m, u + 1, self.a.r, "Away") for u in range(11)))
        self.away_players = self.script.players[-11:]
        
        # --- FIM DA CORREÇÃO ---

        self.players = self.home_players + self.away_players
        self.p_num = len(self.players)

        # Carregar o modelo do adversário, se houver
        self.opponent_model = None
        if opponent_model_path and os.path.exists(opponent_model_path):
            print(f"Carregando modelo do adversário de: {opponent_model_path}")
            self.opponent_model = PPO.load(opponent_model_path)

        # Espaço de Ação e Observação (precisa ser ajustado para suas necessidades)
        self.action_space = gym.spaces.MultiDiscrete([5] * 11)
        self.observation_space = gym.spaces.Box(low=-15, high=15, shape=(self.p_num, 3), dtype=np.float32)

        self.reset()

    def step(self, action):
        # 1. Executar a ação para o time aprendiz
        for i, player in enumerate(self.home_players):
            goal_dir = M.vector_angle((15, 0) - player.world.robot.loc_head_position[:2])
            player.behavior.execute("Basic_Kick", goal_dir)

        # 2. Obter e executar a ação para o time adversário (se houver modelo)
        if self.opponent_model:
            obs = self._get_observation()
            # Precisamos fatiar a observação para pegar apenas a perspectiva do adversário
            # A forma exata de fazer isso depende de como seu espaço de observação é definido
            opponent_obs = obs # Simplificação: o modelo adversário vê tudo
            away_action, _ = self.opponent_model.predict(opponent_obs, deterministic=True)
            for i, player in enumerate(self.away_players):
                 goal_dir = M.vector_angle((-15, 0) - player.world.robot.loc_head_position[:2])
                 player.behavior.execute("Basic_Kick", goal_dir)
        else: # Se não houver modelo, o adversário fica parado
            for i, player in enumerate(self.away_players):
                player.behavior.execute("Zero") # Comando para ficar parado


        # 3. Executar um passo na simulação para todos os jogadores
        self.script.batch_commit_and_send()
        self.script.batch_receive()

        # 4. Calcular a recompensa
        reward, done = self._calculate_reward()
        observation = self._get_observation()
        
        info = {}
        return observation, reward, done, info

    def reset(self):
        print("Reiniciando o ambiente...")
        # Reinicia a posição dos jogadores
        self.script.batch_unofficial_beam((-3, 0, 0.4, 0) for _ in range(self.p_num))
        # Garante que o jogo está em modo 'PlayOn'
        if self.players:
            self.players[0].scom.unofficial_set_play_mode("PlayOn")
        return self._get_observation()

    def _get_observation(self):
        obs = []
        if not self.players:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        for p in self.players:
            # Garante que o world model foi atualizado antes de acessá-lo
            if p.world.time.time > 0:
                 obs.append(p.world.robot.loc_head_position)
            else:
                 obs.append(np.zeros(3)) # Posição padrão se ainda não foi atualizado
        
        # Adiciona a posição da bola
        ball_pos = self.players[0].world.ball.abs_pos[:3] if self.players[0].world.time.time > 0 else np.zeros(3)
        
        # Aqui você deve construir a observação final, por exemplo, concatenando as posições
        # Por enquanto, retornando apenas as posições dos jogadores
        return np.array(obs, dtype=np.float32)


    def _calculate_reward(self):
        reward = 0
        done = False
        
        if not self.home_players:
            return reward, done

        w_home = self.home_players[0].world
        play_mode_str = w_home.game.play_mode_str

        # Penalidade por gol sofrido
        if "Goal_Right" in play_mode_str: # Supondo que 'Home' joga da esquerda para a direita
            print("GOL SOFRIDO!")
            reward -= 10
            done = True

        # Recompensa por gol marcado
        if "Goal_Left" in play_mode_str:
            print("GOL MARCADO!")
            reward += 10
            done = True

        # Recompensa por posse de bola
        if w_home.ball.last_touch_side == 'left':
            reward += 0.1

        # A simulação termina se o tempo acabar (ex: 10 minutos)
        if w_home.time.time >= 600:
            done = True

        return reward, done

    def render(self, mode='human'):
        pass

    def close(self):
        print("Fechando o ambiente e matando o servidor.")
        os.system("pkill -9 rcssserver3d")

class Train:
    def __init__(self, script):
        self.script = script
        self.a = script.args

    def train(self, total_timesteps=50000, opponent_model_path=None, save_path_prefix="gen"):
        env = SelfPlayEnv(self.script, opponent_model_path)
        
        save_path = f"./models/{save_path_prefix}"
        os.makedirs(save_path, exist_ok=True)
        
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./logs/{save_path_prefix}/")
        
        eval_callback = EvalCallback(env, best_model_save_path=save_path,
                                     log_path=save_path, eval_freq=5000,
                                     deterministic=True, render=False)

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save(f"{save_path}/final_model.zip")
        print(f"Modelo final salvo em: {save_path}/final_model.zip")
        print(f"Melhor modelo salvo em: {save_path}/best_model.zip")
        env.close()