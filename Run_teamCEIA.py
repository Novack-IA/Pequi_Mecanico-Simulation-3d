from scripts.commons.Script import Script
script = Script(cpp_builder_unum=1) # Initialize: load config file, parse arguments, build cpp modules
a = script.args

if a.P: # penalty shootout
    from agent.Agent_Penalty import Agent
else: # normal agent
    from agent.CEIA import CEIA

# Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw, Wait for Server, is magmaFatProxy
if a.D: # debug mode
    player = CEIA(a.i, a.p, a.m, a.u, "CEIA", True, True, False, a.F)
else:
    player = CEIA(a.i, a.p, None, a.u, "CEIA", False, False, False, a.F)

while True:
    player.think_and_send()
    player.scom.receive()
