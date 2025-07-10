from agent.Base_AgentCEIA import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
from behaviors.custom.Dribble.Dribble import Dribble
import math
import numpy as np
import json

class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # Load team formation
        with open('config/formation.json', 'r') as f:
            team_formation = json.load(f)

        # Finds the correct robot in the config
        robot_config = next((robot for robot in team_formation if robot['number'] == str(unum)), None)
        if not robot_config:
            raise ValueError(f"Configuration of the {unum} player not found.")

        # define robot type
        robot_type = robot_config['robot_type']
        self.init_pos = robot_config['initial_position'] # initial formation

        # Initialize base agent
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy
        
        self.pos3 = []
        self.pos6 = []
        self.pos9 = []
        self.pos10 = []

        # Instancia o comportamento de drible
        self.dribble = Dribble(self)
        self.dribble.phase = 0  # inicializa a fase do drible

    # ... (beam, move, kick, kick_strong, and fat_proxy methods remain the same) ...

    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance


    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick
        '''
        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()
        
    def kick_strong(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick
        '''
        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Strong_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    # -------------------------------------------------------------------
    # THINK AND SEND - HIGH-LEVEL COORDINATOR
    # -------------------------------------------------------------------
    def think_and_send(self):
        """
        Main loop function: coordinates updating state, making decisions,
        and sending commands.
        """
        self._update_world_state()
        self._handle_game_state()
        self._broadcast_and_send_command()
        self._update_debug_drawings()

    # -------------------------------------------------------------------
    # 1. PREPROCESSING AND STATE UPDATE
    # -------------------------------------------------------------------
    def _update_world_state(self):
        """Gathers and computes all necessary state variables for decision-making."""
        w = self.world
        r = self.world.robot
        
        # Basic world and robot state
        self.my_head_pos_2d = r.loc_head_position[:2]
        self.ball_2d = w.ball_abs_pos[:2]
        ball_vec = self.ball_2d - self.my_head_pos_2d
        self.ball_dir = M.vector_angle(ball_vec)
        self.goal_dir = M.target_abs_angle(self.ball_2d, (15.05, 0))
        self.PM = w.play_mode
        self.PM_GROUP = w.play_mode_group

        # Ball and player distance calculations
        slow_ball_pos = w.get_predicted_ball_pos(0.5)
        teammates_sq_dist = self._calculate_player_distances_to_ball(w.teammates, slow_ball_pos)
        opponents_sq_dist = self._calculate_player_distances_to_ball(w.opponents, slow_ball_pos)
        
        min_teammate_sq_dist = min(teammates_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(min_teammate_sq_dist)
        self.min_opponent_ball_dist = math.sqrt(min(opponents_sq_dist))
        self.active_player_unum = teammates_sq_dist.index(min_teammate_sq_dist) + 1

    def _calculate_player_distances_to_ball(self, players, ball_pos):
        """Calculates squared distances from a list of players to the ball."""
        w = self.world
        distances = []
        for p in players:
            is_valid = p.state_last_update != 0 and \
                       (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and \
                       not p.state_fallen
            if is_valid:
                dist_sq = np.sum((p.state_abs_pos[:2] - ball_pos) ** 2)
                distances.append(dist_sq)
            else:
                distances.append(1000) # Use a large distance for invalid players
        return distances

    # -------------------------------------------------------------------
    # 2. DECISION MAKING
    # -------------------------------------------------------------------
    def _handle_game_state(self):
        """Directs agent behavior based on the current game mode (PM) and agent state."""
        if self.PM == self.world.M_GAME_OVER:
            pass
        elif self.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif self.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True)
        elif self.state == 1 or (self.behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self._handle_getting_up()
        elif self.PM == self.world.M_OUR_KICKOFF:
            self._handle_our_kickoff()
        elif self.PM == self.world.M_THEIR_KICKOFF:
            self.move(self.init_pos, orientation=self.ball_dir)
        elif self.active_player_unum != self.world.robot.unum:
            self._handle_inactive_player_action()
        else: # I am the active player
            self._handle_active_player_action()

    def _handle_getting_up(self):
        """Executes the get-up behavior and updates the agent's state."""
        self.state = 0 if self.behavior.execute("Get_Up") else 1

    def _handle_our_kickoff(self):
        """Logic for when our team has the kickoff."""
        if self.world.robot.unum == 9:
            self.kick(110, 1)
        else:
            self.move(self.init_pos, orientation=self.ball_dir)

    def _handle_inactive_player_action(self):
        """Defines behavior when the agent is not the closest to the ball."""
        r = self.world.robot
        if r.unum == 1: # Goalkeeper
            self.move(self.init_pos, orientation=self.ball_dir)
        else: # Field player
            # Compute formation position based on ball
            new_x = max(0.5, (self.ball_2d[0] + 15) / 15) * (self.init_pos[0] + 15) - 15
            if self.min_teammate_ball_dist < self.min_opponent_ball_dist:
                new_x = min(new_x + 3.5, 13) # Advance if team has possession
            self.move((new_x, self.init_pos[1]), orientation=self.ball_dir, priority_unums=[self.active_player_unum])

    def _handle_active_player_action(self):
        """Directs behavior when the agent is the active player (closest to the ball)."""
        r = self.world.robot
        w = self.world
        
        self.path_manager.draw_options(enable_obstacles=True, enable_path=True, use_team_drawing_channel=True)
        
        enable_pass = (self.PM == w.M_PLAY_ON and self.ball_2d[0] < 6)

        if r.unum == 1:
            self._execute_goalkeeper_active_logic()
        elif self.PM == w.M_OUR_CORNER_KICK:
            self.kick(-np.sign(self.ball_2d[1]) * 95, 10)
        elif r.unum in [6, 9, 10]:
             self._execute_forward_logic(r.unum)
        elif self.min_opponent_ball_dist + 0.5 < self.min_teammate_ball_dist:
            self._execute_defensive_maneuver()
        else: # Default action for other players or fallback
             self._execute_pass_to_closest_forward([6, 9, 10])

        self.path_manager.draw_options(enable_obstacles=False, enable_path=False)

    def _execute_goalkeeper_active_logic(self):
        """Contains all active logic specific to the goalkeeper (unum 1)."""
        if self.PM_GROUP == self.world.MG_THEIR_KICK:
            self.move(self.init_pos, orientation=self.ball_dir)
        elif self.PM_GROUP == self.world.M_OUR_GOAL_KICK:
            if hasattr(self.world, 'pos3'):
                self.pos3 = self.world.pos3.tolist()
                angle = M.target_abs_angle(self.my_head_pos_2d, self.pos3[:2])
                self.kick(angle, 100)
        else: # Default PlayOn action for goalkeeper
             self.state = 0 if self.kick(self.goal_dir, 20, False, True) else 2

    def _execute_forward_logic(self, unum):
        """Determines actions for the forwards (6, 9, 10)."""
        if unum == 6:
            self._execute_pass_to_closest_forward([9, 10])
        elif unum == 9:
            self._execute_striker_action()
        elif unum == 10:
            self._execute_dribbler_action()

    def _execute_pass_to_closest_forward(self, target_unums):
        """Finds the closest teammate from a list and kicks to them."""
        r = self.world.robot
        teammate_positions = {
            6: self.world.pos6.tolist() if hasattr(self.world, 'pos6') else None,
            9: self.world.pos9.tolist() if hasattr(self.world, 'pos9') else None,
            10: self.world.pos10.tolist() if hasattr(self.world, 'pos10') else None,
        }
        
        closest_teammate_unum = -1
        min_dist = float('inf')

        for unum in target_unums:
            pos = teammate_positions.get(unum)
            if pos:
                dist = math.dist(pos[:2], r.loc_head_position[:2])
                if dist < min_dist:
                    min_dist = dist
                    closest_teammate_unum = unum
        
        if closest_teammate_unum != -1:
            target_pos = teammate_positions[closest_teammate_unum]
            angle = M.target_abs_angle(r.loc_head_position[:2], target_pos[:2])
            self.kick(angle, 1000)
        else: # Fallback if no target is found
            self.kick(self.goal_dir, 20)


    def _execute_striker_action(self):
        """Logic for player 9: Shoot if close, otherwise pass to player 10."""
        r = self.world.robot
        if hasattr(self.world, 'pos10'):
            self.pos10 = self.world.pos10.tolist()
            dist_to_10 = math.dist(r.loc_head_position[:2], self.pos10[:2])
            goal_dist = M.distance_point_to_opp_goal(self.ball_2d)

            if goal_dist <= dist_to_10:
                self.kick_strong(self.goal_dir, 1000)
            else:
                angle = M.target_abs_angle(r.loc_head_position[:2], self.pos10[:2])
                self.kick(angle, 1000)
        else:
            self.kick_strong(self.goal_dir, 1000)

    def _execute_dribbler_action(self):
        """Logic for player 10: Dribble towards the goal."""
        if self.PM == self.world.M_PLAY_ON and self.dribble.is_ready():
            dribble_finished = self.dribble.execute(
                reset=False, orientation=None, is_orientation_absolute=True, speed=1, stop=False
            )
            if dribble_finished:
                self.dribble.phase = 0
        else:
            self.kick(self.goal_dir, 5)

    def _execute_defensive_maneuver(self):
        """Positions the agent defensively between the ball and our goal."""
        if self.state == 2:  # Currently kicking, abort it
            self.state = 0 if self.kick(abort=True) else 2
        else:
            # Move between ball and our goal
            target_pos = self.ball_2d + M.normalize_vec((-16, 0) - self.ball_2d) * 0.2
            self.move(target_pos, is_aggressive=True)
            
    # -------------------------------------------------------------------
    # 3. BROADCAST AND SEND
    # -------------------------------------------------------------------
    def _broadcast_and_send_command(self):
        """Broadcasts state and sends the final command to the server."""
        self.radio.broadcast()

        r = self.world.robot
        if self.fat_proxy_cmd is None:  # Normal behavior
            self.scom.commit_and_send(r.get_command())
        else:  # Fat proxy behavior
            self.scom.commit_and_send(self.fat_proxy_cmd.encode())
            self.fat_proxy_cmd = ""

    # -------------------------------------------------------------------
    # 4. DEBUGGING ANNOTATIONS
    # -------------------------------------------------------------------
    def _update_debug_drawings(self):
        """Draws annotations on the monitor for debugging purposes."""
        if self.enable_draw:
            d = self.world.draw
            if self.active_player_unum == self.world.robot.unum:
                slow_ball_pos = self.world.get_predicted_ball_pos(0.5)
                d.point(slow_ball_pos, 3, d.Color.pink, "status", False)
                d.annotation((*self.my_head_pos_2d, 0.6), "I've got it!", d.Color.yellow, "status")
            else:
                d.clear("status")

    # -------------------------------------------------------------------
    # Fat proxy auxiliary methods
    # -------------------------------------------------------------------
    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction - r.imu_torso_orientation ):.2f} 20)"
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d - (-0.1, 0), None, True)
            return False

    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot
        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += f"(proxy dash 100 0 0)"
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg(orientation - r.imu_torso_orientation)
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += f"(proxy dash 0 0 {target_dir:.1f})"
        else:
            self.fat_proxy_cmd += f"(proxy dash 20 0 {target_dir:.1f})"
