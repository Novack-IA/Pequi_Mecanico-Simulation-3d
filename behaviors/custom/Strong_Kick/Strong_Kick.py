from agent.Base_AgentCEIA import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
from math_ops.Math_Ops import Math_Ops as M


class Strong_Kick():

    def __init__(self, base_agent : Base_Agent) -> None:
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.world = base_agent.world
        self.description = "Walk to ball and perform a STRONG kick"
        self.auto_head = True

        r_type = self.world.robot.type
        self.bias_dir = [22,29,26,29,22][self.world.robot.type]
        self.ball_x_limits = ((0.19,0.215), (0.2,0.22), (0.19,0.22), (0.2,0.215), (0.2,0.215))[r_type]
        self.ball_y_limits = ((-0.115,-0.1), (-0.125,-0.095), (-0.12,-0.1), (-0.13,-0.105), (-0.09,-0.06))[r_type]
        self.ball_x_center = (self.ball_x_limits[0] + self.ball_x_limits[1])/2
        self.ball_y_center = (self.ball_y_limits[0] + self.ball_y_limits[1])/2
      
    def execute(self,reset, direction, abort=False) -> bool: # Você pode adicionar mais argumentos
        '''
        Parameters
        ----------
        direction : float
            kick direction relative to field, in degrees
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        '''

        w = self.world
        r = self.world.robot
        b = w.ball_rel_torso_cart_pos
        t = w.time_local_ms
        gait : Step_Generator = self.behavior.get_custom_behavior_object("Walk").env.step_generator

        if reset:
            self.phase = 0
            self.reset_time = t

        if self.phase == 0: 
            biased_dir = M.normalize_deg(direction + self.bias_dir) # adicionar viés para retificar a direção
            ang_diff = abs(M.normalize_deg( biased_dir - r.loc_torso_orientation )) #a reinicialização foi aprendida com loc, não com IMU

            next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                x_ori=biased_dir, x_dev=-self.ball_x_center, y_dev=-self.ball_y_center, torso_ori=biased_dir)
            
            if (w.ball_last_seen > t - w.VISUALSTEP_MS and ang_diff < 5 and       # a bola está visível e alinhada
                self.ball_x_limits[0] < b[0] < self.ball_x_limits[1] and          #a bola está na área de chute (x)
                self.ball_y_limits[0] < b[1] < self.ball_y_limits[1] and          # a bola está na área de chute (y)
                t - w.ball_abs_pos_last_update < 100 and                          # a localização absoluta da bola é recente
                dist_to_final_target < 0.03 and                                   # se a posição absoluta da bola for atualizada
                not gait.state_is_left_active and gait.state_current_ts == 2 and  # a fase da marcha é adequada
                t - self.reset_time > 700): #para evitar chutar imediatamente sem preparação e estabilidade
                #Novack: aumentei o tempo de latência para o chute 600 - 1200
                self.phase += 1

                return self.behavior.execute_sub_behavior("Kick_Motion_strong", True)#novo subcomportamento forte
            else:
                dist = max(0.07, dist_to_final_target)
                reset_walk = reset and self.behavior.previous_behavior != "Walk" # redefinir caminhada se não era o comportamento anterior
                self.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist) # alvo, is_target_abs, ori, is_ori_abs, distância
                return abort #abortar somente se self.phase == 0

        else: #definir parâmetros de chute e executar
            return self.behavior.execute_sub_behavior("Kick_Motion_strong", False)#novo subcomportamento forte

      
    def is_ready(self) -> any: #Você pode adicionar mais argumentos
        ''' Returns True if this behavior is ready to start/continue under current game/robot conditions '''
        return True
