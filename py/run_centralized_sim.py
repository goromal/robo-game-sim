#!/usr/bin/python3

from robo_game_py import GameSim
from src.CentralizedTeam import CentralizedTeam
from src.SimState import SimState
import numpy as np
from math import cos, sin

sim = GameSim()

# Sim parameters
class GameParams:
    def __init__(self):
        self.T = 10.0 # seconds, max total time
        self.dt = 0.05
        self.player_radius = 0.2
        self.puck_radius = 0.175
        self.tau_player= 0.5 
        self.tau_puck = 0.1
        self.input_limit=10.0 
        self.arena_limits_x=10.0 
        self.arena_limits_y=5.0 
        self.winning_score = 4
        self.puck_mass = 0.5
        self.player_mass = 1.0
        self.w_stdev = 0.0 # standard deviation of zero-mean gaussian noise on dynamics inputs
        self.x0_ball = np.array([0.,0.,0.,0.])
        self.log = True
        self.logname = "minimal_game.log"
params = GameParams()
log = params.log
logname = params.logname

# Reset sim (this can be done an arbitrary number of times)
sim.reset(params.dt, params.winning_score, params.x0_ball, params.w_stdev, params.log, params.logname)
sim_state = SimState(sim.run(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)))

# Create two teams
home_team = CentralizedTeam(params, -1, "A", sim_state) # team A defend left goal
away_team = CentralizedTeam(params, 1, "B", sim_state)  # team B defend right goal

# Run the simulator for the full time
t = 0.0
while t < params.T:

    # Compute velocity for each team
    velA1, velA2 = home_team.run(sim_state)
    velB1, velB2 = away_team.run(sim_state)
    
    # commanded velocities for team A and team B
    #velA1 = np.array([cos(t),sin(t)])
    #velA2 = np.array([cos(t),sin(t)])
    #velB1 = np.array([cos(t),sin(t)])
    #velB2 = np.array([cos(t),sin(t)])

    # run the simulator, returns vector with all sim info
    sim_state = SimState(sim.run(velA1, velA2, velB1, velB2))
    #print(sim_state.transpose())

    t += params.dt


