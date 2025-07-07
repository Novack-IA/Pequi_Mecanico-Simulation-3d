from scripts.commons.Script import Script
script = Script() # Inicializar: carregar arquivo de configuração, analisar argumentos, construir módulos cpp
a = script.args

from agent.CEIA import Agent

# Argumentos: IP do servidor, porta do agente, porta do monitor, número do uniforme, nome da equipe, habilitar log, habilitar desenho
team_args = ((a.i, a.p, a.m, u, a.t, True, True) for u in range(1,12))
script.batch_create(Agent,team_args)

while True:
    script.batch_execute_agent()
    script.batch_receive()
