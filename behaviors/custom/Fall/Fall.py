from agent.Base_AgentCEIA import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Neural_Network import run_mlp
import pickle, numpy as np

class Fall():

    def __init__(self, base_agent : Base_Agent) -> None:
        self.world = base_agent.world
        self.description = "Fall example"
        self.auto_head = False

        with open(M.get_active_directory("/behaviors/custom/Fall/fall.pkl"), 'rb') as f:
            self.model = pickle.load(f)

        self.action_size = len(self.model[-1][0]) #extraído do tamanho do último viés da camada da rede neural
        self.obs = np.zeros(self.action_size+1, np.float32)

        self.controllable_joints = min(self.world.robot.no_of_joints, self.action_size) # compatibilidade entre diferentes tipos de robôs

    def observe(self):
        r = self.world.robot
        
        for i in range(self.action_size):
            self.obs[i] = r.joints_position[i] / 100 #normalização de escala ingênua

        self.obs[self.action_size] = r.cheat_abs_pos[2] # head.z (alternativa: r.loc_head_z)
      
    def execute(self,reset) -> bool:
        self.observe()
        action = run_mlp(self.obs, self.model) 
        
        self.world.robot.set_joints_target_position_direct( # cometer ações:
            slice(self.controllable_joints), # atuar em articulações treinadas
            action*10,                       #aumentar as ações para motivar a exploração precoce
            harmonize=False                  # não faz sentido harmonizar ações se os alvos mudam a cada passo
        )

        return self.world.robot.loc_head_z < 0.15 #terminado quando a altura da cabeça for < 0,15 m
    def is_ready(self) -> any:
        ''' Returns True if this behavior is ready to start/continue under current game/robot conditions '''
        return True
