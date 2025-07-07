from agent.Base_AgentCEIA import Base_Agent
from collections import deque
import numpy as np

class Get_Up():

    def __init__(self, base_agent : Base_Agent) -> None:
        self.behavior = base_agent.behavior
        self.world = base_agent.world
        self.description = "Get Up using the most appropriate skills"
        self.auto_head = False
        self.MIN_HEIGHT = 0.3 #valor mínimo para a altura da cabeça
        self.MAX_INCLIN = 50  #inclinação máxima do tronco em graus
        self.STABILITY_THRESHOLD = 4

    def reset(self):
        self.state = 0
        self.gyro_queue = deque(maxlen=self.STABILITY_THRESHOLD)
        self.watchdog = 0 # quando o jogador tem o bug de tremor, ele nunca fica estável o suficiente para se levantar

    def execute(self,reset):

        r = self.world.robot
        execute_sub_behavior = self.behavior.execute_sub_behavior
        
        if reset:
            self.reset()

        if self.state == 0: # Estado 0: ir para a pose "Zero"

            self.watchdog += 1
            self.gyro_queue.append( max(abs(r.gyro)) ) # registrar os últimos valores de STABILITY_THRESHOLD

            # avançar para o próximo estado se o comportamento estiver completo e o robô estiver estável
            if (execute_sub_behavior("Zero",None) and len(self.gyro_queue) == self.STABILITY_THRESHOLD 
                and all(g < 10 for g in self.gyro_queue)) or self.watchdog > 100:

                #determinar como se levantar
                if r.acc[0] < -4 and abs(r.acc[1]) < 2 and abs(r.acc[2]) < 3:
                    execute_sub_behavior("Get_Up_Front", True) # redefinir comportamento
                    self.state = 1
                elif r.acc[0] > 4 and abs(r.acc[1]) < 2 and abs(r.acc[2]) < 3:
                    execute_sub_behavior("Get_Up_Back", True) # redefinir comportamento
                    self.state = 2
                elif r.acc[2] > 8: # à prova de falhas se a visão não estiver atualizada: se a pose for 'Zero' e o torso estiver ereto, o robô já está em pé
                    return True
                else:
                    execute_sub_behavior("Flip", True) # redefinir comportamento
                    self.state = 3

        elif self.state == 1:
            if execute_sub_behavior("Get_Up_Front", False):
                return True
        elif self.state == 2:
            if execute_sub_behavior("Get_Up_Back", False):
                return True
        elif self.state == 3:
            if execute_sub_behavior("Flip", False):
                self.reset()

        return False
        

    def is_ready(self):
        ''' Returns True if the Get Up behavior is ready (= robot is down) '''
        r = self.world.robot
        # verifique se z < 5 e magnitude de aceleração > 8 e qualquer indicador visual diz que caímos
        return r.acc[2] < 5 and np.dot(r.acc,r.acc) > 64 and (r.loc_head_z < self.MIN_HEIGHT or r.imu_torso_inclination > self.MAX_INCLIN)
