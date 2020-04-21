#!/usr/bin/python3

from robo_game_py import GameSim
import numpy as np
from math import cos, sin

sim = GameSim()

# Sim parameters
T = 20.0 # seconds, max total time
dt = 0.05
winning_score = 4
x0_ball = np.array([0.,0.,0.,0.])
log = True
logname = "minimal_game.log"

# Reset sim (this can be done an arbitrary number of times)
sim.reset(dt, winning_score, x0_ball, log, logname)

# Run the simulator for the full time
t = 0.0
while t < T:
    # commanded velocities for team A and team B
    velA1 = np.array([cos(t),sin(t)])
    velA2 = np.array([cos(t),sin(t)])
    velB1 = np.array([cos(t),sin(t)])
    velB2 = np.array([cos(t),sin(t)])

    # run the simulator, returns vector with all sim info
    sim_state = sim.run(velA1, velA2, velB1, velB2)
    print(sim_state.transpose())

    t += dt
