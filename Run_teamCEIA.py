from scripts.commons.Script import Script
script = Script(cpp_builder_unum=1) # Inicializar: carregar arquivo de configuração, analisar argumentos, construir módulos cpp
a = script.args

if a.P: # disputa de pênaltis
    from agent.CEIA_Penalty import Agent
else: # agente normal
    from agent.CEIA import Agent

# Argumentos: IP do servidor, porta do agente, porta do monitor, número do uniforme, nome da equipe, habilitar log, habilitar draw, aguardar servidor, é magmaFatProxy
if a.D: # modo de depuração
    player = Agent(a.i, a.p, a.m, a.u, "CEIA", True, True, False, a.F)
else:
    player = Agent(a.i, a.p, None, a.u, "CEIA", False, False, False, a.F)

while True:
    player.think_and_send()
    player.scom.receive()
