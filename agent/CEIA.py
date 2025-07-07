from agent.Base_AgentCEIA import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
from behaviors.custom.Dribble.Dribble import Dribble
import math
import numpy as np

team = [
    {
        'number': '1',
        'initial_position': [-14, 0],
        'robot_type': 0
    },
    {
        'number': '2',
        'initial_position': [-11, -3],
        'robot_type': 1
    },
    {
        'number': '3',
        'initial_position': [-11, 3],
        'robot_type': 1
    },
    {
        'number': '4',
        'initial_position': [-9, 0],
        'robot_type': 1
    },
    {
        'number': '5',
        'initial_position': [-6, -4],
        'robot_type': 2
    },
    {
        'number': '6',
        'initial_position': [-5, 0],
        'robot_type': 2
    },
    {
        'number': '7',
        'initial_position': [-4, -2],
        'robot_type': 2
    },
    {
        'number': '8',
        'initial_position': [-6, 4],
        'robot_type': 4
    },
    {
        'number': '9',
        'initial_position': [-1, -2.5],
        'robot_type': 3
    },
    {
        'number': '10',
        'initial_position': [-1, 2.5],
        'robot_type': 3
    },
    {
        'number': '11',
        'initial_position': [-4, 2],
        'robot_type': 4
    },
]

class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # definir tipo de robô
        robot = team[unum-1]
        robot_type = robot['robot_type']
        self.init_pos = robot['initial_position'] # formação inicial

        # Inicializar agente base
        # Argumentos: IP do servidor, porta do agente, porta do monitor, número do uniforme, tipo de robô, nome da equipe, habilitar log, habilitar draw, correção do modo de jogo, esperar pelo servidor, ouvir retorno de chamada
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  #0-Normal, 1-Levantar, 2-Chutar
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) #parâmetros de caminhada filtrados para proxy de gordura
        
        self.pos3 = []
        self.pos6 = []
        self.pos9 = []
        self.pos10 = []

        # Instância de comportamento de drible
        self.dribble = Dribble(self)
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # parâmetros de caminhada filtrados para proxy de gordura
        
        self.pos3 = []
        self.pos6 = []
        self.pos9 = []
        self.pos10 = []

        # Instância de comportamento de drible
        self.dribble = Dribble(self)
        self.dribble.phase = 0  # inicializa a fase do drible

    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] #copiar lista de posições
        self.state = 0

        #Evite o círculo central movendo o jogador para trás
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # feixe para posição inicial, coordenada de face (0,0)
        else:
            if self.fat_proxy_cmd is None: # comportamento normal
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) #redefinir caminhada de proxy gordo


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # comportamento de proxy gordo
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Argumentos: alvo, is_target_abs, ori, is_ori_abs, distância



    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # comportamento normal
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) #Basic_Kick não tem controle de distância de chute
        else: # comportamento de proxy gordo
            return self.fat_proxy_kick()
        

    def kick_strong(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: #comportamento normal
            return self.behavior.execute("Strong_Kick", self.kick_direction, abort) #Basic_Kick não tem controle de distância de chute
        else: # comportamento de proxy gordo
            return self.fat_proxy_kick()


    def think_and_send(self):
        w = self.world
        r = self.world.robot  
        my_head_pos_2d = r.loc_head_position[:2]
        my_ori = r.imu_torso_orientation
        ball_2d = w.ball_abs_pos[:2]
        ball_vec = ball_2d - my_head_pos_2d
        ball_dir = M.vector_angle(ball_vec)
        ball_dist = np.linalg.norm(ball_vec)
        ball_sq_dist = ball_dist * ball_dist # para comparações mais rápidas
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])
        behavior = self.behavior
        goal_dir = M.target_abs_angle(ball_2d,(15.05,0))
        path_draw_options = self.path_manager.draw_options
        PM = w.play_mode
        PM_GROUP = w.play_mode_group
        possession_threshold_1 = 0.1
        #--------------------------------------- 1. Preprocessing

        slow_ball_pos = w.get_predicted_ball_pos(0.5) #posição futura prevista da bola 2D quando a velocidade da bola <= 0,5 m/s

        # lista de distâncias quadradas entre companheiros de equipe (incluindo ele mesmo) e a bola lenta (a distância quadrada é definida como 1000 em algumas condições)
        teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)  # distância quadrada entre o companheiro de equipe e a bola
                                  if p.state_last_update != 0 and (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
                                  else 1000 # forçar grande distância se o companheiro de equipe não existir, ou suas informações de estado não forem recentes (360 ms), ou se ele tiver caído
                                  for p in w.teammates ]

        # lista de distâncias quadradas entre oponentes e bola lenta (a distância quadrada é definida como 1000 em algumas condições)
        opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)  # distância quadrada entre o companheiro de equipe e a bola
                                  if p.state_last_update != 0 and w.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
                                  else 1000 # forçar grande distância se o oponente não existir, ou se suas informações de estado não forem recentes (360 ms), ou se ele tiver caído
                                  for p in w.opponents ]

        min_teammate_ball_sq_dist = min(teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(min_teammate_ball_sq_dist)   # distância entre a bola e o companheiro de equipe mais próximo
        self.min_opponent_ball_dist = math.sqrt(min(opponents_ball_sq_dist)) # distância entre a bola e o oponente mais próximo

        active_player_unum = teammates_ball_sq_dist.index(min_teammate_ball_sq_dist) + 1

        #--------------------------------------- 2. Decida a ação

        if PM == w.M_GAME_OVER:
            pass
        elif PM_GROUP == w.MG_ACTIVE_BEAM:
            self.beam()
        elif PM_GROUP == w.MG_PASSIVE_BEAM:
            self.beam(True) # evite o círculo central
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1 #retornar ao estado normal se o comportamento de levantar tiver terminado
        elif PM == w.M_OUR_KICKOFF:
            if r.unum == 9:
                self.kick(110, 1) # não há necessidade de alterar o estado quando o PM não estiver em Play On
            else:
                self.move(self.init_pos, orientation=ball_dir) # andar no lugar
        elif PM == w.M_THEIR_KICKOFF:
            self.move(self.init_pos, orientation=ball_dir) # andar no lugar
        elif active_player_unum != r.unum: # Eu não sou o jogador ativo
            if r.unum == 1: #Eu sou o goleiro
                self.move(self.init_pos, orientation=ball_dir) # andar no lugar 
            else:
                #calcular a posição básica da formação com base na posição da bola
                new_x = max(0.5,(ball_2d[0]+15)/15) * (self.init_pos[0]+15) - 15
                if self.min_teammate_ball_dist < self.min_opponent_ball_dist:
                    new_x = min(new_x + 3.5, 13) # avançar se a equipe tiver posse de bola
                self.move((new_x,self.init_pos[1]), orientation=ball_dir, priority_unums=[active_player_unum])

        else: # Eu sou o jogador ativo          
            path_draw_options(enable_obstacles=True, enable_path=True, use_team_drawing_channel=True) # habilitar desenhos de caminho para o jogador ativo (ignorado se self.enable_draw for False)
            enable_pass_command = (PM == w.M_PLAY_ON and ball_2d[0]<6)

            if r.unum == 1 and PM_GROUP == w.MG_THEIR_KICK: # goleiro durante seu chute
                self.move(self.init_pos, orientation=ball_dir) #andar no lugar
            elif r.unum == 1 and PM_GROUP == w.M_OUR_GOAL_KICK:
                if hasattr(self.world, 'pos3'):
                    self.pos3 = self.world.pos3.tolist()
                
                angle = M.target_abs_angle(r.loc_head_position[:2], self.pos3[:2])
                self.kick(angle, 100)
            if PM == w.M_OUR_CORNER_KICK:
                self.kick( -np.sign(ball_2d[1])*95, 10) #chutar a bola para o espaço em frente ao gol do adversário
                #não há necessidade de alterar o estado quando o PM não estiver em Play On
            if r.unum not in [6, 9, 10]:
                if hasattr(self.world, 'pos6'):
                    self.pos6 = self.world.pos6.tolist()

                if hasattr(self.world, 'pos9'):
                    self.pos9 = self.world.pos9.tolist()

                if hasattr(self.world, 'pos10'):
                    self.pos10 = self.world.pos10.tolist()


                dist6 = math.dist(self.pos6[:2], r.loc_head_position[:2])
                dist9 = math.dist(self.pos9[:2], r.loc_head_position[:2])
                dist10 = math.dist(self.pos10[:2], r.loc_head_position[:2])
                distancias = (dist6, dist9, dist10)
                print('Distancias (6): ', distancias)
                p_proximo = distancias.index(min(distancias))

                if p_proximo == 0:
                    angle = M.target_abs_angle(r.loc_head_position[:2], self.pos6[:2])
                    self.kick(angle, 1000)           
                elif p_proximo == 1:
                    angle = M.target_abs_angle(r.loc_head_position[:2], self.pos9[:2])
                    self.kick(angle, 1000)           
                elif p_proximo == 2:
                    angle = M.target_abs_angle(r.loc_head_position[:2], self.pos10[:2])
                    self.kick(angle, 1000)           

            elif r.unum == 6:
                if hasattr(self.world, 'pos9'):
                    self.pos9 = self.world.pos9.tolist()

                if hasattr(self.world, 'pos10'):
                    self.pos10 = self.world.pos10.tolist()

                dist9 = math.dist(self.pos9[:2], r.loc_head_position[:2])
                dist10 = math.dist(self.pos10[:2], r.loc_head_position[:2])
                distancias = (dist9, dist10)
                print('DISTANCIAS (todos): ', distancias)
                p_proximo = distancias.index(min(distancias))

                if p_proximo == 0:
                    angle = M.target_abs_angle(r.loc_head_position[:2], self.pos9[:2])
                    self.kick(angle, 1000)           
                elif p_proximo == 1:
                    angle = M.target_abs_angle(r.loc_head_position[:2], self.pos10[:2])
                    self.kick(angle, 1000)
            elif r.unum == 9:
                if hasattr(self.world, 'pos9'):
                    self.pos9 = self.world.pos9.tolist()

                if hasattr(self.world, 'pos10'):
                    self.pos10 = self.world.pos10.tolist()

                goal_dist = M.distance_point_to_opp_goal(ball_2d[:2])
                dist_9_10 = math.sqrt((self.pos9[0] - self.pos10[0])**2 + (self.pos9[1] - self.pos10[1])**2)

                if goal_dist <= dist_9_10:
                    self.kick(goal_dir, 1000)
                else:
                    angle = M.target_abs_angle(r.loc_head_position[:2], self.pos10[:2])
                    self.kick(angle, 1000)                        
            elif r.unum == 10:
                if PM == w.M_PLAY_ON and self.dribble.is_ready():
                    # Aqui, podemos definir as configurações de drible.
                    # Por exemplo, se nenhum ângulo foi especificado, o drible irá buscar uma orientação para o gol adversário.
                    reset = False  # ou alguma condição que indique que o drible deve ser reiniciado
                    orientation = None  # drible para o gol
                    is_orientation_absolute = True
                    speed = 1  # velocidade máxima
                    stop = False
                    dribble_finished = self.dribble.execute(reset, orientation, is_orientation_absolute, speed, stop)
                    if dribble_finished:
                        # Se o dribble terminou, podemos reiniciar a fase ou tomar outra ação.
                        self.dribble.phase = 0
                else:
                    self.kick(goal_dir, 5)
            elif self.min_opponent_ball_dist + 0.5 < self.min_teammate_ball_dist: # defender se o adversário estiver consideravelmente mais perto da bola
                if self.state == 2: # comprometer-se a chutar enquanto aborta
                    self.state = 0 if self.kick(abort=True) else 2
                else: #mover-me em direção à bola, mas posicionar-me entre a bola e o nosso objetivo
                    self.move(slow_ball_pos + M.normalize_vec((-16,0) - slow_ball_pos) * 0.2, is_aggressive=True)
            else:
                self.state = 0 if self.kick(goal_dir,20,False,enable_pass_command) else 2

            path_draw_options(enable_obstacles=False, enable_path=False) # desativar desenhos de caminho

        #--------------------------------------- 3. Transmissão
        self.radio.broadcast()

        #--------------------------------------- 4. Enviar para o servidor
        if self.fat_proxy_cmd is None: # comportamento normal
            self.scom.commit_and_send( r.get_command() )
        else: # comportamento de proxy gordo
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""

        #----------------------anotações para depuração
        if self.enable_draw: 
            d = w.draw
            if active_player_unum == r.unum:
                d.point(slow_ball_pos, 3, d.Color.pink, "status", False) # posição futura prevista da bola 2D quando a velocidade da bola <= 0,5 m/s
                d.point(w.ball_2d_pred_pos[-1], 5, d.Color.pink, "status", False) # previsão da última bola
                d.annotation((*my_head_pos_2d, 0.6), "I've got it!" , d.Color.yellow, "status")
            else:
                d.clear("status")




    #--------------------------------------- Métodos auxiliares de proxy gordo


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # argumentos de chute de proxy gordo: potência [0,10]; ângulo horizontal relativo [-180,180]; ângulo vertical [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) #redefinir caminhada de proxy gordo
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignorar obstáculos
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")