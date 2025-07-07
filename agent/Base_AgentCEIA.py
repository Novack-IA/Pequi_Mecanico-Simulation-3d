from abc import abstractmethod
from behaviors.Behavior import Behavior
from communication.Radio import Radio
from communication.Server_Comm import Server_Comm
from communication.World_Parser import World_Parser
from logs.Logger import Logger
from math_ops.Inverse_Kinematics import Inverse_Kinematics
from world.commons.Path_Manager import Path_Manager
from world.World import World

class Base_Agent():
    all_agents = []

    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int, robot_type:int, team_name:str, enable_log:bool=True,
                  enable_draw:bool=True, apply_play_mode_correction:bool=True, wait_for_server:bool=True, hear_callback=None) -> None:

        self.radio = None # hear_message pode ser chamado durante a instanciação do Server_Comm
        self.logger = Logger(enable_log, f"{team_name}_{unum}")
        self.world = World(robot_type, team_name, unum, apply_play_mode_correction, enable_draw, self.logger, host)
        self.world_parser = World_Parser(self.world, self.hear_message if hear_callback is None else hear_callback)
        self.scom = Server_Comm(host,agent_port,monitor_port,unum,robot_type,team_name,self.world_parser,self.world,Base_Agent.all_agents,wait_for_server)
        self.inv_kinematics = Inverse_Kinematics(self.world.robot)
        self.behavior = Behavior(self)
        self.path_manager = Path_Manager(self.world)
        self.radio = Radio(self.world, self.scom.commit_announcement)
        self.behavior.create_behaviors()
        Base_Agent.all_agents.append(self)

    @abstractmethod
    def think_and_send(self):
        pass

    def hear_message(self, msg:bytearray, direction, timestamp:float) -> None:
        if direction != "self" and self.radio is not None:
            self.radio.receive(msg)

    def terminate(self):
        # feche o soquete do monitor compartilhado se este for o último agente neste tópico
        self.scom.close(close_monitor_socket=(len(Base_Agent.all_agents)==1))
        Base_Agent.all_agents.remove(self)

    @staticmethod
    def terminate_all():
        for o in Base_Agent.all_agents:
            o.scom.close(True) # feche o soquete do monitor compartilhado, se existir
        Base_Agent.all_agents = []

