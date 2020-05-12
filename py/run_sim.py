#!/usr/bin/python3

from robo_game_py import GameSim
from src.ClassicalTeam import ClassicalTeam
from src.SimState import SimState
import numpy as np
from math import cos, sin
from src.CBF import CBF

sim = GameSim()

# Sim parameters
class GameParams:
    def __init__(self):
        self.T = 50.0 # seconds, max total time
        self.dt = 0.05
        self.player_radius = 0.2
        self.puck_radius = 0.175
        self.tau_player=0.5
        self.tau_puck = 0.1 # 0.1 was too difficult to bouncing off the wall and move to goal
        self.player_mass = 1.0
        self.puck_mass = 0.5
        self.input_limit= 10.0
        self.arena_limits_x=10.0 
        self.arena_limits_y=5.0
        self.goal_height = 1.0
        self.winning_score = 4
        self.x0_ball = np.array([-1, 0.,0.,0.]) # np.array([0.,0.,0.,0.])
        self.log = True
        self.logname = "minimal_game.log"
        self.gamma = 1.
        self.safety_radius = 2*self.player_radius
        self.barrier_gain = 0.1
params = GameParams()
log = params.log
logname = params.logname

# Reset sim (this can be done an arbitrary number of times)
sim.reset(params.dt, params.winning_score, params.x0_ball, 0, params.log, params.logname)
sim_state = SimState(sim.run(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)))

# Create two teams
home_team = ClassicalTeam(params, -1, "A", sim_state) # team A defend left goal
away_team = ClassicalTeam(params, 1, "B", sim_state)  # team B defend right goal

# Run the simulator for the full time
t = 0.0

# Initialize centralized CBF
centralized_CBF = CBF(params, params.safety_radius, params.barrier_gain)

while t < params.T:

    # Compute velocity for each team
    velA1, velA2 = home_team.run(sim_state)
    velB1, velB2 = away_team.run(sim_state)
    
    # commanded velocities for team A and team B
    #velA1 = np.array([cos(t),sin(t)])
    #velA2 = np.array([cos(t),sin(t)])
    #velB1 = np.array([cos(t),sin(t)])
    #velB2 = np.array([cos(t),sin(t)])

    # Centralized safety controller
    u_nominal = [velA1, velA2, velB1, velB2]
    velocities = [sim_state.get_player_vel("A", 1),
                  sim_state.get_player_vel("A", 2),
                  sim_state.get_player_vel("B", 1),
                  sim_state.get_player_vel("B", 2)
                  ]
    positions = [sim_state.get_player_pos("A", 1),
                  sim_state.get_player_pos("A", 2),
                  sim_state.get_player_pos("B", 1),
                  sim_state.get_player_pos("B", 2)
                  ]
    velA1, velA2, velB1, velB2 = centralized_CBF.get_centralized_safe_control(u_nominal, velocities, positions)

    # run the simulator, returns vector with all sim info
    sim_state = SimState(sim.run(velA1, velA2, velB1, velB2))
    # print(sim_state.transpose())

    t += params.dt


