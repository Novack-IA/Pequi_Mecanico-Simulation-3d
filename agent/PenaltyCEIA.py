from agent.Base_AgentCEIA import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import numpy as np
import random


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # definir tipo de robô
        robot_type = 0 if unum == 1 else 4 # suponha que o goleiro use o uniforme número 1 e o chutador use qualquer outro número

        # Inicializar agente base
        # Argumentos: IP do servidor, porta do agente, porta do monitor, número do uniforme, tipo de robô, nome da equipe, habilitar log, habilitar draw, correção do modo de jogo, esperar pelo servidor, ouvir retorno de chamada
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, False, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Levantando-se, 2-Mergulho para a esquerda, 3-Mergulho para a direita, 4-Espere

        self.kick_dir = 0 # direção do chute
        self.reset_kick = True # quando Verdadeiro, uma nova direção de chute aleatória é gerada
        

    def think_and_send(self):
        w = self.world
        r = self.world.robot 
        my_head_pos_2d = r.loc_head_position[:2]
        my_ori = r.imu_torso_orientation
        ball_2d = w.ball_abs_pos[:2]
        ball_vec = ball_2d - my_head_pos_2d
        ball_dir = M.vector_angle(ball_vec)
        ball_dist = np.linalg.norm(ball_vec)
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])
        behavior = self.behavior
        PM = w.play_mode

        #--------------------------------------- 1. Decide action

        if PM in [w.M_BEFORE_KICKOFF, w.M_THEIR_GOAL, w.M_OUR_GOAL]: # feixe para a posição inicial e aguarde
            self.state = 0
            self.reset_kick = True
            pos = (-14,0) if r.unum == 1 else (4.9,0)
            if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or behavior.is_ready("Get_Up"):
                self.scom.commit_beam(pos, 0) # feixe para a posição inicial
            else:
                behavior.execute("Zero_Bent_Knees") # espere
        elif self.state == 2: # mergulhar para a esquerda
            self.state = 4 if behavior.execute("Dive_Left") else 2  # alterar estado para esperar após a habilidade terminar
        elif self.state == 3: # mergulhe para a direita
            self.state = 4 if behavior.execute("Dive_Right") else 3 # alterar estado para esperar após a habilidade terminar
        elif self.state == 4: # esperar (após o mergulho ou durante o chute adversário)
            pass
        elif self.state == 1 or behavior.is_ready("Get_Up"): #se levantando ou caindo
            self.state = 0 if behavior.execute("Get_Up") else 1 #retornar ao estado normal se o comportamento de levantar tiver terminado
        elif PM == w.M_OUR_KICKOFF and r.unum == 1 or PM == w.M_THEIR_KICKOFF and r.unum != 1:
            self.state = 4 # espere até o próximo feixe
        elif r.unum == 1: # goleiro
            y_coordinate = np.clip(ball_2d[1], -1.1, 1.1)
            behavior.execute("Walk", (-14,y_coordinate), True, 0, True, None) #Argumentos: alvo, is_target_abs, ori, is_ori_abs, distância
            if ball_2d[0] < -10: 
                self.state = 2 if ball_2d[1] > 0 else 3 # mergulhar para defender
        else: # chutador
            if PM == w.M_OUR_KICKOFF and ball_2d[0] > 5: # verifique a posição da bola para ter certeza de que a vejo
                if self.reset_kick: 
                    self.kick_dir = random.choice([-7.5,7.5]) 
                    self.reset_kick = False
                behavior.execute("Basic_Kick", self.kick_dir)
            else:
                behavior.execute("Zero_Bent_Knees") # espere

        #--------------------------------------- 2. Transmissão
        self.radio.broadcast()

        #--------------------------------------- 3. Enviar para o servidor
        self.scom.commit_and_send( r.get_command() )

        #---------------------- anotações para depuração
        if self.enable_draw: 
            d = w.draw
            if r.unum == 1:
                d.annotation((*my_head_pos_2d, 0.8), "Goalkeeper" , d.Color.yellow, "status")
            else:
                d.annotation((*my_head_pos_2d, 0.8), "Kicker" , d.Color.yellow, "status")
                if PM == w.M_OUR_KICKOFF: # desenhe uma seta para indicar a direção do chute
                    d.arrow(ball_2d, ball_2d + 5*M.vector_from_angle(self.kick_dir), 0.4, 3, d.Color.cyan_light, "Target")


