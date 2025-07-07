from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.CEIA import Agent


class BaseStrategy(ABC):
    """
    Classe base abstrata para definir a estratégia de um jogador.

    Cada estratégia de papel (como Goleiro, Atacante, etc.) deve herdar
    desta classe e implementar o método `execute`.
    """

    def __init__(self, agent: 'Agent'):
        """
        Inicializa a estratégia base.

        Args:
            agent (Agent): A instância do agente que esta estratégia irá controlar.
        """
        self.agent = agent
        self.world = agent.world
        self.robot = agent.world.robot

    @abstractmethod
    def execute(self) -> None:
        """
        Executa a lógica principal para o papel do jogador.

        Este método deve ser implementado por todas as classes de estratégia concretas.
        Ele contém a lógica de tomada de decisão para o agente com base em seu papel
        no estado atual do jogo.
        """
        pass