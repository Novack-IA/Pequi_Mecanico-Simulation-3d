from agent.Base_AgentCEIA import Base_Agent
from behaviors.custom.Dribble.Env import Env
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Neural_Network import run_mlp
import numpy as np
import pickle


class Dribble():

    def __init__(self, base_agent : Base_Agent) -> None:
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.world = base_agent.world
        self.description = "RL dribble"
        self.auto_head = True
        self.env = Env(base_agent, 0.9 if self.world.robot.type == 3 else 1.2)

        with open(M.get_active_directory([
            "/behaviors/custom/Dribble/dribble_R0.pkl",
            "/behaviors/custom/Dribble/dribble_R1.pkl",
            "/behaviors/custom/Dribble/dribble_R2.pkl",
            "/behaviors/custom/Dribble/dribble_R3.pkl",
            "/behaviors/custom/Dribble/dribble_R4.pkl"
            ][self.world.robot.type]), 'rb') as f:
            self.model = pickle.load(f)

    def define_approach_orientation(self):

        w = self.world
        b = w.ball_abs_pos[:2]
        me = w.robot.loc_head_position[:2]

        self.approach_orientation = None

        MARGIN = 0.8 # margem de segurança (se a bola estiver próxima dos limites do campo por essa quantidade, a orientação de aproximação é considerada)
        M90 = 90/MARGIN # variável auxiliar
        DEV = 25 # quando estiver em cima das linhas laterais ou finais, a direção de aproximação desvia-se dessa linha por essa quantidade
        MDEV = (90+DEV)/MARGIN # variável auxiliar

        a1 = -180 # início do intervalo de ângulo (rotação no sentido anti-horário)
        a2 = 180  # fim do intervalo de ângulo (rotação no sentido anti-horário)

        if b[1] < -10 + MARGIN:
            if b[0] < -15 + MARGIN:
                a1 = DEV - M90 * (b[1]+10)
                a2 = 90 - DEV + M90 * (b[0]+15)
            elif b[0] > 15 - MARGIN:
                a1 = 90 + DEV - M90 * (15-b[0])
                a2 = 180 - DEV + M90 * (b[1]+10)
            else:
                a1 = DEV - MDEV * (b[1]+10)
                a2 = 180 - DEV + MDEV * (b[1]+10)
        elif b[1] > 10 - MARGIN:
            if b[0] < -15 + MARGIN:
                a1 = -90 + DEV - M90 * (b[0]+15)
                a2 = -DEV + M90 * (10-b[1])
            elif b[0] > 15 - MARGIN:
                a1 = 180 + DEV - M90 * (10-b[1])
                a2 = 270 - DEV + M90 * (15-b[0])
            else:
                a1 = -180 + DEV - MDEV * (10-b[1])
                a2 = -DEV + MDEV * (10-b[1])
        elif b[0] < -15 + MARGIN:
            a1 = -90 + DEV - MDEV * (b[0]+15)
            a2 = 90 - DEV + MDEV * (b[0]+15)
        elif b[0] > 15 - MARGIN and abs(b[1]) > 1.2:
            a1 = 90 + DEV - MDEV * (15-b[0])
            a2 = 270 - DEV + MDEV * (15-b[0])


        cad = M.vector_angle(b - me) # direção de aproximação atual

        a1 = M.normalize_deg(a1)
        a2 = M.normalize_deg(a2)

        if a1<a2: 
            if a1 <= cad <= a2:  
                return # a orientação da abordagem atual está dentro da faixa aceita
        else:
            if a1 <= cad or cad <= a2:
                return # a orientação da abordagem atual está dentro da faixa aceita

        a1_diff = abs(M.normalize_deg(a1 - cad))
        a2_diff = abs(M.normalize_deg(a2 - cad))

        self.approach_orientation = a1 if a1_diff < a2_diff else a2  # orientação normalizada fixa


    def execute(self, reset, orientation, is_orientation_absolute, speed=1, stop=False):
        '''
        Parameters
        ----------
        orientation : float
            absolute or relative orientation of torso (relative to imu_torso_orientation), in degrees
            set to None to dribble towards the opponent's goal (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        speed : float
            speed from 0 to 1 (scale is not linear)
        stop : bool
            return True immediately if walking, wind down if dribbling, and return True when possible
        '''
        w = self.world
        r = self.world.robot
        me = r.loc_head_position[:2]
        b = w.ball_abs_pos[:2]
        b_rel = w.ball_rel_torso_cart_pos[:2]
        b_dist = np.linalg.norm(b-me)
        behavior = self.behavior
        reset_dribble = False
        lost_ball = (w.ball_last_seen <= w.time_local_ms - w.VISUALSTEP_MS) or np.linalg.norm(b_rel)>0.4

        if reset:
            self.phase = 0
            if behavior.previous_behavior == "Push_RL" and 0<b_rel[0]<0.25 and abs(b_rel[1])<0.07:
                self.phase = 1
                reset_dribble = True

        if self.phase == 0: # caminhar para a bola
            reset_walk = reset and behavior.previous_behavior != "Walk" and behavior.previous_behavior != "Push_RL" # redefinir caminhada se não era o comportamento anterior

            #------------------------ 1. Decida se é necessária uma melhor orientação de aproximação (quando a bola está quase fora dos limites)
            if reset or b_dist > 0.4: # pare de definir a orientação após se aproximar da bola para evitar ruído
                self.define_approach_orientation()

            #------------------------ 2A.É necessária uma melhor orientação de aproximação (a bola está quase fora dos limites)
            if self.approach_orientation is not None:
                next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                    x_ori=self.approach_orientation, x_dev=-0.24, torso_ori=self.approach_orientation, safety_margin=0.4)

                if b_rel[0]<0.26 and b_rel[0]>0.18 and abs(b_rel[1])<0.04 and w.ball_is_visible: #pronto para começar a driblar
                    self.phase += 1
                    reset_dribble = True
                else:
                    dist = max(0.08, dist_to_final_target*0.7)
                    behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist) # alvo, is_target_abs, ori, is_ori_abs, distância

            #------------------------ 2B. Não é necessária uma melhor orientação de aproximação, mas o robô não consegue ver a bola
            elif w.time_local_ms - w.ball_last_seen > 200: #caminhe até o alvo absoluto se a bola não for vista
                abs_ori = M.vector_angle( b - me )
                behavior.execute_sub_behavior("Walk", reset_walk, b, True, abs_ori, True, None) # alvo, is_target_abs, ori, is_ori_abs, distância
                
            #------------------------ 2C. Não é necessária uma melhor orientação de aproximação e o robô pode ver a bola
            else: # caminhar até o alvo relativo   
                if 0.18<b_rel[0]<0.25 and abs(b_rel[1])<0.05 and w.ball_is_visible: #pronto para começar a driblar
                    self.phase += 1
                    reset_dribble = True
                else:
                    rel_target = b_rel+(-0.23,0)    # alvo relativo é um círculo (centro: bola, raio: 0,23 m)
                    rel_ori = M.vector_angle(b_rel) # orientação relativa da bola, NÃO do alvo!
                    dist = max(0.08, np.linalg.norm(rel_target)*0.7) # abordagem lenta
                    behavior.execute_sub_behavior("Walk", reset_walk, rel_target, False, rel_ori, False, dist) # alvo, is_target_abs, ori, is_ori_abs, distância

            if stop:
                return True

        if self.phase == 1 and (stop or (b_dist > 0.5 and lost_ball)): # volte a caminhar
            self.phase += 1
        elif self.phase == 1: # Drible
#----------------------- 1. Defina os parâmetros do drible
            self.env.dribble_speed = speed
        
            #Os valores de orientação relativa são diminuídos para evitar ultrapassagem
            if orientation is None:
                if b[0] < 0: # driblar para os lados
                    if b[1] > 0:
                        dribble_target = (15,5)
                    else:
                        dribble_target = (15,-5)
                else:
                    dribble_target = None # ir para o gol
                self.env.dribble_rel_orientation = self.path_manager.get_dribble_path(optional_2d_target=dribble_target)[1]
            elif is_orientation_absolute:
                self.env.dribble_rel_orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            else:
                self.env.dribble_rel_orientation = float(orientation) #copiar se numpy float

            #------------------------ 2.Executar comportamento
            obs = self.env.observe(reset_dribble)
            action = run_mlp(obs, self.model)   
            self.env.execute(action)
        
        #diminuir o ritmo do drible e então reiniciar a fase
        if self.phase > 1:
            WIND_DOWN_STEPS = 60
            #------------------------ 1. Definir parâmetros de redução de drible
            self.env.dribble_speed = 1 - self.phase/WIND_DOWN_STEPS 
            self.env.dribble_rel_orientation = 0

            #------------------------ 2. Executar comportamento
            obs = self.env.observe(reset_dribble, virtual_ball=True)
            action = run_mlp(obs, self.model)   
            self.env.execute(action)

            #------------------------ 3.Redefinir comportamento 
            self.phase += 1
            if self.phase >= WIND_DOWN_STEPS - 5:
                self.phase = 0
                return True

            
        return False

    def is_ready(self):
        ''' Returns True if this behavior is ready to start/continue under current game/robot conditions '''
        return True