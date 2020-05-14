from robo_game_py import GameSim
import numpy as np

sim = GameSim()

# Sim parameters
class GameParams:
    def __init__(self):
        self.T = 20.0 # seconds, max total time
        self.dt = 0.05
        self.player_radius = 0.2
        self.puck_radius = 0.175
        self.tau_player=0.5
        self.input_limit=10.0
        self.arena_limits_x=10.0
        self.arena_limits_y=5.0
        self.winning_score = 4
        self.x0_ball = np.array([0.,0.,0.,0.])
        self.log = True
        self.logname = "minimal_game.log"
params = GameParams()
log = params.log
logname = params.logname

# Reset sim (this can be done an arbitrary number of times)
sim.reset(params.dt, params.winning_score, params.x0_ball, 0.0, params.log, params.logname)
sim.run(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))

velA1list = list()
velA2list = list()
velB1list = list()
velB2list = list()
with open('A1.txt','r') as A1f, open('A2.txt','r') as A2f, open('B1.txt','r') as B1f, open('B2.txt','r') as B2f:
    for line in A1f:
        v1 = float(line.split()[0])
        v2 = float(line.split()[1])
        velA1list.append(np.array([v1,v2]))
    for line in A2f:
        v1 = float(line.split()[0])
        v2 = float(line.split()[1])
        velA2list.append(np.array([v1,v2]))
    for line in B1f:
        v1 = float(line.split()[0])
        v2 = float(line.split()[1])
        velB1list.append(np.array([v1,v2]))
    for line in B2f:
        v1 = float(line.split()[0])
        v2 = float(line.split()[1])
        velB2list.append(np.array([v1,v2]))

# Run simulator
t = 0.0
i = 0
while t < params.T:
    velA1 = velA1list[i]
    velA2 = velA2list[i]
    velB1 = velB1list[i]
    velB2 = velB2list[i]
    sim.run(velA1, velA2, velB1, velB2)
    i += 1
    t += params.dt
