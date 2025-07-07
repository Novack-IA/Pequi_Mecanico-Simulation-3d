from agent.Base_AgentCEIA import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

class Env():
    def __init__(self, base_agent : Base_Agent, step_width) -> None:

        self.world = base_agent.world
        self.ik = base_agent.inv_kinematics
        
        # Espaço de estado 
        self.obs = np.zeros(76, np.float32)
        
        # Padrões de comportamento de etapas
        self.STEP_DUR = 8
        self.STEP_Z_SPAN = 0.02
        self.STEP_Z_MAX = 0.70

        # IK 
        r = self.world.robot
        nao_specs = self.ik.NAO_SPECS
        self.leg_length = nao_specs[1] + nao_specs[3] # altura da parte superior da perna + altura da parte inferior da perna
        feet_y_dev = nao_specs[0] * step_width # passo mais largo
        sample_time = r.STEPTIME
        max_ankle_z = nao_specs[5]

        self.step_generator = Step_Generator(feet_y_dev, sample_time, max_ankle_z)
        self.DEFAULT_ARMS = np.array([-90,-90,8,8,90,90,70,70],np.float32)

        self.dribble_rel_orientation = None # em relação a imu_torso_orientation (em graus)
        self.dribble_speed = 1


    def observe(self, init=False, virtual_ball=False):

        w = self.world
        r = self.world.robot

        if init: #redefinir variáveis
            self.step_counter = 0
            self.act = np.zeros(16, np.float32) # variável de memória

        # index       observação             normalização ingênua
        self.obs[0] = min(self.step_counter,12*8) /100  # contador simples: 0,1,2,3...
        self.obs[1] = r.loc_head_z                *3    # coordenada z (torso)
        self.obs[2] = r.loc_head_z_vel            /2    # velocidade z (torso)  
        self.obs[3] = r.imu_torso_roll            /15   # rotação absoluta do tronco em graus
        self.obs[4] = r.imu_torso_pitch           /15   # inclinação absoluta do tronco em graus
        self.obs[5:8] = r.gyro                    /100  # giroscópio
        self.obs[8:11] = r.acc                    /10   # acelerômetro

        self.obs[11:17] = r.frp.get('lf', np.zeros(6)) * (10,10,10,0.01,0.01,0.01) #  pé esquerdo: ponto de origem relativo (p) e vetor de força (f) -> (px,py,pz,fx,fy,fz)*
        self.obs[17:23] = r.frp.get('rf', np.zeros(6)) * (10,10,10,0.01,0.01,0.01) #  pé direito: ponto de origem relativo (p) e vetor de força (f) -> (px,py,pz,fx,fy,fz)*
        # *se o pé não estiver tocando o chão, então (px=0,py=0,pz=0,fx=0,fy=0,fz=0)

        self.obs[23:43] = r.joints_position[2:22] /100    # posição de todas as articulações, exceto cabeça e dedos dos pés (para robô tipo 4)
        self.obs[43:63] = r.joints_speed[2:22]    /6.1395 # velocidade de todas as articulações, exceto cabeça e dedos dos pés (para robô tipo 4)

        '''
        Expected observations for walking state:
        Time step        R  0   1   2   3   4   5   6   7   0
        Progress         1  0 .14 .28 .43 .57 .71 .86   1   0
        Left leg active  T  F   F   F   F   F   F   F   F   T
        '''

        if init: # os parâmetros de caminhada referem-se aos últimos parâmetros em vigor (após uma reinicialização, eles não têm sentido)
            self.obs[63] = 1 #progresso da etapa
            self.obs[64] = 1 # 1 se a perna esquerda estiver ativa
            self.obs[65] = 0 # 1 se a perna direita estiver ativa
            self.obs[66] = 0
        else:
            self.obs[63] = self.step_generator.external_progress #progresso da etapa
            self.obs[64] = float(self.step_generator.state_is_left_active)     #1 se a perna esquerda estiver ativa
            self.obs[65] = float(not self.step_generator.state_is_left_active) #1 se a perna direita estiver ativa
            self.obs[66] = math.sin(self.step_generator.state_current_ts / self.step_generator.ts_per_step * math.pi)

        # Bola
        ball_rel_hip_center = self.ik.torso_to_hip_transform(w.ball_rel_torso_cart_pos)
        ball_dist_hip_center = np.linalg.norm( ball_rel_hip_center )

        if init:
            self.obs[67:70] = (0,0,0) # A velocidade inicial é 0
        elif w.ball_is_visible:
            self.obs[67:70] = (ball_rel_hip_center - self.obs[70:73]) * 10  # Velocidade da bola em relação ao ponto médio do tornozelo
            
        self.obs[70:73] = ball_rel_hip_center #Posição da bola em relação ao quadril
        self.obs[73] = ball_dist_hip_center * 2

        if virtual_ball: # simular a bola entre os pés do robô
            self.obs[67:74] = (0,0,0,0.05,0,-0.175,0.36)

        '''
        Create internal target with a smoother variation
        '''

        MAX_ROTATION_DIFF = 20 # diferença máxima (graus) por passo visual
        MAX_ROTATION_DIST = 80

        if init:
            self.internal_rel_orientation = 0
            self.internal_target_vel = 0
            self.gym_last_internal_abs_ori = r.imu_torso_orientation #para fins de treinamento (recompensa)

       
        #---------------------------------------------------------------- calcular alvo interno
        
        if w.vision_is_up_to_date:

            previous_internal_rel_orientation = np.copy(self.internal_rel_orientation)

            internal_ori_diff =  np.clip( M.normalize_deg( self.dribble_rel_orientation - self.internal_rel_orientation ), -MAX_ROTATION_DIFF, MAX_ROTATION_DIFF)
            self.internal_rel_orientation = np.clip(M.normalize_deg( self.internal_rel_orientation + internal_ori_diff ), -MAX_ROTATION_DIST, MAX_ROTATION_DIST)

            # Observações
            self.internal_target_vel = self.internal_rel_orientation - previous_internal_rel_orientation

            self.gym_last_internal_abs_ori = self.internal_rel_orientation + r.imu_torso_orientation

        #----------------------------------------------------------------- Observações
        
        self.obs[74] = self.internal_rel_orientation / MAX_ROTATION_DIST
        self.obs[75] = self.internal_target_vel / MAX_ROTATION_DIFF

        return self.obs


    def execute_ik(self, l_pos, l_rot, r_pos, r_rot):
        r = self.world.robot
        #Aplique IK em cada perna + Defina alvos articulares
          
        #Perna esquerda
        indices, self.values_l, error_codes = self.ik.leg(l_pos, l_rot, True, dynamic_pose=False)

        r.set_joints_target_position_direct(indices, self.values_l, harmonize=False)

        # Perna direita
        indices, self.values_r, error_codes = self.ik.leg(r_pos, r_rot, False, dynamic_pose=False)

        r.set_joints_target_position_direct(indices, self.values_r, harmonize=False)


    def execute(self, action):
        
        r = self.world.robot

       # Ações:
# 0,1,2 posição do tornozelo esquerdo
# 3,4,5 posição do tornozelo direito
# 6,7,8 rotação do pé esquerdo
# 9,10,11 rotação do pé direito
# 12,13 arremesso do braço esquerdo/direito
# 14,15 rotação do braço esquerdo/direito

       # média móvel exponencial
        self.act = 0.85 * self.act + 0.15 * action * 0.7 * 0.95 * self.dribble_speed

        # execute o comportamento Step para extrair as posições de destino de cada perna (substituiremos essas metas)
        lfy,lfz,rfy,rfz = self.step_generator.get_target_positions(self.step_counter == 0, self.STEP_DUR, self.STEP_Z_SPAN, self.leg_length * self.STEP_Z_MAX)

        # Perna IK
        a = self.act
        l_ankle_pos = (a[0]*0.025-0.01, a[1]*0.01 + lfy, a[2]*0.01 + lfz)
        r_ankle_pos = (a[3]*0.025-0.01, a[4]*0.01 + rfy, a[5]*0.01 + rfz)
        l_foot_rot = a[6:9]  * (2,2,3)
        r_foot_rot = a[9:12] * (2,2,3)

        # Limite a guinada/inclinação da perna (e adicione inclinação)
        l_foot_rot[2] = max(0,l_foot_rot[2] + 18.3)
        r_foot_rot[2] = min(0,r_foot_rot[2] - 18.3)

        # Ações de braços
        arms = np.copy(self.DEFAULT_ARMS) # pose de braços padrão
        arms[0:4] += a[12:16]*4 # arremesso + rotação dos braços

        # Defina posições-alvo
        self.execute_ik(l_ankle_pos, l_foot_rot, r_ankle_pos, r_foot_rot)           # pernas
        r.set_joints_target_position_direct( slice(14,22), arms, harmonize=False )  # braços

        self.step_counter += 1