import math
import numpy as np
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from agent.Base_AgentCEIA import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
from behaviors.custom.Dribble.Dribble import Dribble

# =============================================================================
#
#                      THE ART OF ROBOT SOCCER
#
#                          By: Sun Tzu Futbol
#
#      "The supreme art of war is to subdue the enemy without fighting.
#     The supreme art of soccer is to score before the enemy knows
#                      the battle has begun."
#
# =============================================================================


# =============================================================================
# === 1. THE INSTRUMENTS OF WAR (Core Data Structures)
# =============================================================================

@dataclass
class Action:
    """A command from the mind to the body. The embodiment of intent."""
    type: str  # 'MOVE', 'KICK', 'KICK_STRONG', 'DRIBBLE', 'FEINT'
    target: tuple[float, float] = (0, 0)
    orientation: float = 0.0
    power: float = 100.0
    message: str = "" # A whisper of intent to our comrades

# =============================================================================
# === 2. THE MIND OF THE GENERAL (Strategy & Tactics)
# =============================================================================

class StrategyManager:
    """The General's tent. Here, the grand strategy is forged from the chaos of battle."""
    def __init__(self):
        self.posture = 'PROBE'  # Start by probing the enemy's defenses.

    def update_posture(self, world, ball_pos):
        """Observe the flow of battle and adapt the grand strategy."""
        game_time_sec = world.time_local_ms / 1000.0
        time_remaining = 600 - game_time_sec
        score_diff = world.our_score - world.their_score

        # Late in the game, desperation or cunning dictates the flow.
        if time_remaining < 150:
            if score_diff < 0:
                self.posture = 'OVERWHELM' # All-out attack. Victory or glorious defeat.
            elif score_diff > 0:
                self.posture = 'FEIGNED_RETREAT' # Lure them into a final, fatal trap.
            else:
                self.posture = 'PROBE'
            return self.posture

        # If we are being dominated, we must adapt.
        if ball_pos[0] < -8.0 and self.posture != 'FEIGNED_RETREAT':
             self.posture = 'PROBE' # We have lost the advantage. Regroup.

        return self.posture

class Tactics:
    """The scrolls of cunning. The methods of artifice and opportunity."""
    @staticmethod
    def find_artful_pass(world, my_pos):
        """Find not the closest, but the most advantageous pass."""
        best_target = None
        max_score = -float('inf')

        for teammate in world.teammates:
            if teammate.is_self or teammate.state_last_update == 0:
                continue

            teammate_pos = teammate.state_abs_pos[:2]
            
            # An artful pass favors a forward who is open.
            receiver_openness = min([math.dist(teammate_pos, o.state_abs_pos[:2]) for o in world.opponents if o.state_last_update != 0] or [100])
            forward_advantage = teammate_pos[0] - my_pos[0]
            
            score = (0.6 * receiver_openness) + (0.4 * forward_advantage)
            
            if score > max_score:
                max_score = score
                best_target = teammate_pos
        
        return best_target

    @staticmethod
    def find_through_ball_opportunity(world, my_pos):
        """A deceptive pass that splits the defense, creating chaos."""
        for teammate in world.teammates:
            if teammate.is_self or teammate.state_last_update == 0: continue
            
            teammate_pos = teammate.state_abs_pos[:2]
            path_vector = teammate_pos - my_pos
            path_midpoint = my_pos + path_vector * 0.5

            # Is the path to our teammate clear of immediate threats?
            path_clear = all(math.dist(path_midpoint, o.state_abs_pos[:2]) > 2.0 for o in world.opponents if o.state_last_update != 0)

            if path_clear and teammate_pos[0] > my_pos[0]:
                # The target is not the player, but the space in front of them.
                return teammate_pos + M.normalize_vec(path_vector) * 2.0
        return None

    @staticmethod
    def calculate_overwhelm_vector(my_pos, world):
        """Find the weakest point in the enemy line to apply overwhelming force."""
        goal_pos = np.array([15.0, 0.0])
        # Find the opponent closest to the direct path to the goal
        closest_opp_to_path = None
        min_dist = float('inf')

        for opp in world.opponents:
            if opp.state_last_update != 0:
                dist = np.linalg.norm(np.cross(goal_pos - my_pos, my_pos - opp.state_abs_pos[:2])) / np.linalg.norm(goal_pos - my_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_opp_to_path = opp.state_abs_pos[:2]
        
        if closest_opp_to_path is not None and min_dist < 3.0:
            # Attack the shoulder of the closest defender, not the center.
            return goal_pos + np.array([0, 3.0 * np.sign(closest_opp_to_path[1] * -1)])
        return goal_pos

# =============================================================================
# === 3. THE FORMATIONS OF THE ARMY (Role-Based Logic)
# =============================================================================

class Role(ABC):
    """A soldier's duty, defined by the needs of the battle."""
    def __init__(self, agent):
        self.agent = agent
        self.world = agent.world
        self.my_pos = agent.my_head_pos_2d

    @abstractmethod
    def decide(self) -> Action:
        pass

class TheAnchor(Role): # Goalkeeper
    """The unmovable mountain. The last line of defiance."""
    def decide(self) -> Action:
        ball_pos = self.world.ball_abs_pos[:2]
        if np.linalg.norm(ball_pos - self.my_pos) < 1.2:
            # A brutal clearance. No subtlety here.
            return Action('KICK_STRONG', orientation=M.vector_angle(ball_pos - self.my_pos))
        
        # Cunning Distribution: If posture is PROBE, start a controlled attack.
        if self.agent.strategic_posture == 'PROBE':
            nearby_defender = Tactics.find_artful_pass(self.world, self.my_pos)
            if nearby_defender is not None and nearby_defender[0] < -8.0:
                return Action('KICK', target=nearby_defender, power=30, message="Begin the flow.")

        # Default: Hold the line using the triangle strategy.
        target_pos = Tactics.get_goalkeeper_triangle_position(ball_pos)
        return Action('MOVE', target=target_pos, orientation=M.vector_angle(ball_pos - self.my_pos))

class TheSpearhead(Role): # Attacker
    """The point of the spear. The embodiment of brutal offense."""
    def decide(self) -> Action:
        # In OVERWHELM posture, we are a force of nature.
        if self.agent.strategic_posture == 'OVERWHELM':
            target = Tactics.calculate_overwhelm_vector(self.my_pos, self.world)
            return Action('DRIBBLE', orientation=M.target_abs_angle(self.my_pos, target), message="Unleash the storm!")
        
        # Look for a deceptive through-ball.
        through_ball_target = Tactics.find_through_ball_opportunity(self.world, self.my_pos)
        if through_ball_target is not None:
            return Action('KICK_STRONG', target=through_ball_target, message="Deception!")

        # If no other option, dribble with cunning.
        dribble_target = Tactics.get_avoidance_dribble_target(self.my_pos, self.world)
        return Action('DRIBBLE', orientation=M.target_abs_angle(self.my_pos, dribble_target))

class TheShadow(Role): # Supporter / Midfielder
    """The unseen blade. Creates opportunities through movement and guile."""
    def decide(self) -> Action:
        # My primary weapon is movement without the ball, to break the enemy's shape.
        # This logic would be in an off-ball role, but here we decide what to do with the ball.
        pass_target = Tactics.find_artful_pass(self.world, self.my_pos)
        if pass_target is not None:
            return Action('KICK', target=pass_target, message="The path is open.")
        
        # If no pass, a probing dribble.
        return Action('DRIBBLE', orientation=M.target_abs_angle(self.my_pos, (15.0, 0.0)))

class TheInterceptor(Role): # Defender
    """The patient hunter. Turns the enemy's attack into our opportunity."""
    def decide(self) -> Action:
        # When we win the ball, we do not just clear it. We begin the counter-strike.
        if self.agent.strategic_posture == 'FEIGNED_RETREAT':
             # We have sprung the trap. Find the furthest forward player.
             counter_target = Tactics.find_artful_pass(self.world, self.my_pos)
             if counter_target is not None:
                 return Action('KICK_STRONG', target=counter_target, message="The trap is sprung!")
        
        # Default: A safe, intelligent pass to a Shadow.
        pass_target = Tactics.find_artful_pass(self.world, self.my_pos)
        if pass_target is not None:
            return Action('KICK', target=pass_target)
        
        return Action('DRIBBLE', orientation=0) # Dribble upfield safely.

# =============================================================================
# === 4. THE GENERAL ON THE FIELD (Main Agent Class)
# =============================================================================

class Agent(Base_AgentCEIA):
    """The vessel of the General's will. It perceives, decides, and acts."""
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        with open('config/formation.json', 'r') as f:
            team_formation = json.load(f)
        robot_config = next((robot for robot in team_formation if robot['number'] == str(unum)), None)
        super().__init__(host, agent_port, monitor_port, unum, robot_config['robot_type'], team_name, enable_log, enable_draw, True, wait_for_server, None)
        self.init_pos = robot_config['initial_position']
        self.strategy_manager = StrategyManager()
        self.dribble = Dribble(self)
        self.strategic_posture = 'PROBE'

    def think_and_send(self):
        """The cycle of war: Observe, Orient, Decide, Act."""
        self._update_world_state()
        self.strategic_posture = self.strategy_manager.update_posture(self.world, self.ball_2d)
        role = self._assign_role()
        action = role.decide()
        self._execute_action(action)
        self._broadcast_and_send_command(action.message)

    def _assign_role(self) -> Role:
        """To know your soldiers is to know victory."""
        unum = self.world.robot.unum
        if unum == 1: return TheAnchor(self)
        
        if self.active_player_unum == unum:
            if unum in [9, 10]: return TheSpearhead(self)
            else: return TheShadow(self)
        
        # Off-ball logic
        # A true Sun Tzu would have many off-ball roles. For now, they are Interceptors.
        return TheInterceptor(self)

    def _execute_action(self, action: Action):
        """Translate will into action."""
        target_orientation = action.orientation
        if action.target is not None and np.any(action.target):
            target_orientation = M.target_abs_angle(self.my_head_pos_2d, action.target)

        if action.type == 'MOVE':
            self.move(target_2d=action.target, orientation=target_orientation)
        elif action.type == 'KICK':
            self.kick(kick_direction=target_orientation, kick_distance=action.power)
        elif action.type == 'KICK_STRONG':
            self.kick_strong(kick_direction=target_orientation, kick_distance=action.power)
        elif action.type == 'DRIBBLE':
            self.kick(kick_direction=target_orientation, kick_distance=5)

    def _update_world_state(self):
        """To know the battlefield is to be halfway to victory."""
        w = self.world
        self.my_head_pos_2d = w.robot.loc_head_position[:2]
        self.ball_2d = w.ball_abs_pos[:2]
        
        teammates_sq_dist = [np.sum((p.state_abs_pos[:2] - self.ball_2d) ** 2) if p.state_last_update != 0 else 1000 for p in w.teammates]
        self.active_player_unum = np.argmin(teammates_sq_dist) + 1

    def _broadcast_and_send_command(self, message=""):
        """Whispers on the wind carry the seeds of victory or defeat."""
        if message: self.radio.say(message)
        else: self.radio.broadcast()
        self.scom.commit_and_send(self.world.robot.get_command())
