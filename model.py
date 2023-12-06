from simulations.sim_constrained import *

simulation = Env("data/useCase_2_stages.json",visualise=False)

while simulation.alive:
    
    #print(list(simulation.possible_actions.items()))

    action_order = random.choice(list(simulation.possible_actions.keys()))
    action_resource = random.choice(simulation.possible_actions[action_order])
    
    #print("Action is taken: " , (action_order,action_resource))
    #print()
    simulation.step((action_order,action_resource))