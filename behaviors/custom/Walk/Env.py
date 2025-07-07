from agent.Base_AgentCEIA import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np


class Env():
    def __init__(self, base_agent : Base_Agent) -> None:

        self.world = base_agent.world
        self.ik = base_agent.inv_kinematics
        
        #Espaço de estado
        self.obs = np.zeros(63, np.float32)
        
        # Padrões de comportamento de etapas
        self.STEP_DUR = 8
        self.STEP_Z_SPAN = 0.02
        self.STEP_Z_MAX = 0.70

        # IK 
        nao_specs = self.ik.NAO_SPECS
        self.leg_length = nao_specs[1] + nao_specs[3] # altura da parte superior da perna + altura da parte inferior da perna
        feet_y_dev = nao_specs[0] * 1.12 # passo mais largo
        sample_time = self.world.robot.STEPTIME
        max_ankle_z = nao_specs[5]

        self.step_generator = Step_Generator(feet_y_dev, sample_time, max_ankle_z)
        self.DEFAULT_ARMS = np.array([-90,-90,8,8,90,90,70,70],np.float32)

        self.walk_rel_orientation = None
        self.walk_rel_target = None
        self.walk_distance = None


    def observe(self, init=False):

        r = self.world.robot

        if init: # redefinir variáveis
            self.act = np.zeros(16, np.float32) # variável de memória
            self.step_counter = 0

        # índice          observação             normalização ingênua
        self.obs[0] = min(self.step_counter,15*8) /100  # contador simples: 0,1,2,3...
        self.obs[1] = r.loc_head_z                *3   # coordenada z (torso)
        self.obs[2] = r.loc_head_z_vel            /2   #velocidade z (torso)  
        self.obs[3] = r.imu_torso_roll            /15   # rotação absoluta do tronco em graus
        self.obs[4] = r.imu_torso_pitch           /15   #inclinação absoluta do tronco em graus
        self.obs[5:8] = r.gyro                    /100  # giroscópio
        self.obs[8:11] = r.acc                    /10   # acelerômetro

        self.obs[11:17] = r.frp.get('lf', np.zeros(6)) * (10,10,10,0.01,0.01,0.01) #  pé esquerdo: ponto de origem relativo (p) e vetor de força (f) -> (px,py,pz,fx,fy,fz)*
        self.obs[17:23] = r.frp.get('rf', np.zeros(6)) * (10,10,10,0.01,0.01,0.01) # pé direito: ponto de origem relativo (p) e vetor de força (f) -> (px,py,pz,fx,fy,fz)*
        # *se o pé não estiver tocando o chão, então (px=0,py=0,pz=0,fx=0,fy=0,fz=0)

        # Articulações: Cinemática para frente para tornozelos + rotação dos pés + braços (arremesso + rolamento)
        rel_lankle = self.ik.get_body_part_pos_relative_to_hip("lankle") # posição do tornozelo em relação ao centro de ambas as articulações do quadril
        rel_rankle = self.ik.get_body_part_pos_relative_to_hip("rankle") # posição do tornozelo em relação ao centro de ambas as articulações do quadril
        lf = r.head_to_body_part_transform("torso", r.body_parts['lfoot'].transform ) # transformação do pé em relação ao tronco
        rf = r.head_to_body_part_transform("torso", r.body_parts['rfoot'].transform ) # transformação do pé em relação ao tronco
        lf_rot_rel_torso = np.array( [lf.get_roll_deg(), lf.get_pitch_deg(), lf.get_yaw_deg()] ) #rotação do pé em relação ao tronco
        rf_rot_rel_torso = np.array( [rf.get_roll_deg(), rf.get_pitch_deg(), rf.get_yaw_deg()] ) #rotação do pé em relação ao tronco

        # pose
        self.obs[23:26] = rel_lankle * (8,8,5)
        self.obs[26:29] = rel_rankle * (8,8,5)
        self.obs[29:32] = lf_rot_rel_torso / 20
        self.obs[32:35] = rf_rot_rel_torso / 20
        self.obs[35:39] = r.joints_position[14:18] /100 #braços (arremesso + rolamento)

        # velocidade
        self.obs[39:55] = r.joints_target_last_speed[2:18] # previsões == última ação

        '''
        Expected observations for walking state:
        Time step        R  0   1   2   3   4   5   6   7   0
        Progress         1  0 .14 .28 .43 .57 .71 .86   1   0
        Left leg active  T  F   F   F   F   F   F   F   F   T
        '''

        if init: #os parâmetros de caminhada referem-se aos últimos parâmetros em vigor (após uma reinicialização, eles não têm sentido)
            self.obs[55] = 1 # progresso da etapa
            self.obs[56] = 1 # 1 se a perna esquerda estiver ativa
            self.obs[57] = 0 # 1 se a perna direita estiver ativa
        else:
            self.obs[55] = self.step_generator.external_progress # progresso da etapa
            self.obs[56] = float(self.step_generator.state_is_left_active)     # 1 se a perna esquerda estiver ativa
            self.obs[57] = float(not self.step_generator.state_is_left_active) # 1 se a perna direita estiver ativa

        '''
        Create internal target with a smoother variation
        '''

        MAX_LINEAR_DIST = 0.5
        MAX_LINEAR_DIFF = 0.014 # diferença máxima (metros) por passo
        MAX_ROTATION_DIFF = 1.6 # diferença máxima (graus) por passo
        MAX_ROTATION_DIST = 45


        if init:      
            self.internal_rel_orientation = 0
            self.internal_target = np.zeros(2)

        previous_internal_target = np.copy(self.internal_target)
       
        #---------------------------------------------------------------- calcular alvo linear interno
        
        rel_raw_target_size = np.linalg.norm(self.walk_rel_target)

        if rel_raw_target_size == 0:
            rel_target = self.walk_rel_target
        else:
            rel_target = self.walk_rel_target / rel_raw_target_size * min(self.walk_distance, MAX_LINEAR_DIST)
       
        internal_diff = rel_target - self.internal_target
        internal_diff_size = np.linalg.norm(internal_diff)

        if internal_diff_size > MAX_LINEAR_DIFF:
            self.internal_target += internal_diff * (MAX_LINEAR_DIFF / internal_diff_size)
        else:
            self.internal_target[:] = rel_target

        #---------------------------------------------------------------- calcular meta de rotação interna
        internal_ori_diff =  np.clip( M.normalize_deg( self.walk_rel_orientation - self.internal_rel_orientation ), -MAX_ROTATION_DIFF, MAX_ROTATION_DIFF)
        self.internal_rel_orientation = np.clip(M.normalize_deg( self.internal_rel_orientation + internal_ori_diff ), -MAX_ROTATION_DIST, MAX_ROTATION_DIST)

        #-----------------------------------------------------------------observações
        
        internal_target_vel = self.internal_target - previous_internal_target

        self.obs[58] = self.internal_target[0] / MAX_LINEAR_DIST
        self.obs[59] = self.internal_target[1] / MAX_LINEAR_DIST
        self.obs[60] = self.internal_rel_orientation / MAX_ROTATION_DIST
        self.obs[61] = internal_target_vel[0] / MAX_LINEAR_DIFF
        self.obs[62] = internal_target_vel[0] / MAX_LINEAR_DIFF

        return self.obs


    def execute_ik(self, l_pos, l_rot, r_pos, r_rot):
        r = self.world.robot
        # Aplique IK em cada perna + Defina alvos articulares
          
        # Perna esquerda
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

        internal_dist = np.linalg.norm( self.internal_target )
        action_mult = 1 if internal_dist > 0.2 else (0.7/0.2) * internal_dist + 0.3

        # média móvel exponencial
        self.act = 0.8 * self.act + 0.2 * action * action_mult * 0.7
        
        #execute o comportamento Step para extrair as posições de destino de cada perna (substituiremos essas metas)
        lfy,lfz,rfy,rfz = self.step_generator.get_target_positions(self.step_counter == 0, self.STEP_DUR, self.STEP_Z_SPAN, self.leg_length * self.STEP_Z_MAX)


        # Perna IK
        a = self.act
        l_ankle_pos = (a[0]*0.02, max(0.01,  a[1]*0.02 + lfy), a[2]*0.01 + lfz) # limite y para evitar autocolisão
        r_ankle_pos = (a[3]*0.02, min(a[4]*0.02 + rfy, -0.01), a[5]*0.01 + rfz) # limite y para evitar autocolisão
        l_foot_rot = a[6:9]  * (3,3,5)
        r_foot_rot = a[9:12] * (3,3,5)

        # Limite de guinada/inclinação da perna
        l_foot_rot[2] = max(0,l_foot_rot[2] + 7)
        r_foot_rot[2] = min(0,r_foot_rot[2] - 7)

        # Ações de braços
        arms = np.copy(self.DEFAULT_ARMS) # pose de braços padrão
        arm_swing = math.sin(self.step_generator.state_current_ts / self.STEP_DUR * math.pi) * 6
        inv = 1 if self.step_generator.state_is_left_active else -1
        arms[0:4] += a[12:16]*4 + (-arm_swing*inv,arm_swing*inv,0,0) # arremesso + rotação dos braços

        # Defina posições-alvo
        self.execute_ik(l_ankle_pos, l_foot_rot, r_ankle_pos, r_foot_rot)           # pernas
        r.set_joints_target_position_direct( slice(14,22), arms, harmonize=False )  # braços

        self.step_counter += 1