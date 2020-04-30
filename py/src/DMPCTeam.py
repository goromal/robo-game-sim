import numpy as np
from src.DMPCPlayer import DMPCPlayer
from pydrake.all import LinearSystem

class DMPCParams(object):
    def __init__(self):
        self.sys = None             # continuous state space system
        self.minT = None            # minimum length of time step (s)
        self.maxT = None            # maximum length of time step (s)
        self.N = None               # number of time steps in horizon
        self.D = None               # neighborhood radius
        self.k_lambda = None        # obstacle potential function gain
        self.d_safe = None          # maximum reach of potential function
        self.k_tang = None          # gain on tangential potential
        self.d_band = None          # width of tangential band
        self.Omega_n_u = None       # cost weight matrix for control effort
        self.Omega_N_max = None     # default cost weight matrix for final state error
        self.Omega_N_min = None     # minimum cost weight matrix for final state error

class StrategyParams(object):
    def __init__(self):
        self.v_thresh = None
        self.d_goalie_offense = None
        self.d_goalie_defense = None
        self.v_hit = None

class DMPCTeam(object):
    def __init__(self, sim_params, field, name):
        # Global sim parameters
        self.name = name
        self.sim_params = sim_params
        self.field = field

        # Declare controller parameters
        self.controller_params = DMPCParams()
        self.controller_params.N = 20
        self.controller_params.minT = self.sim_params.dt / self.controller_params.N
        self.controller_params.maxT = 5.0 / self.controller_params.N
        self.controller_params.sys = LinearSystem(np.array([[0, 0,                               1,                               0],
                                                            [0, 0,                               0,                               1],
                                                            [0, 0, -1.0/self.sim_params.tau_player,                               0],
                                                            [0, 0,                               0, -1.0/self.sim_params.tau_player]]),
                                                  np.array([[                             0,                              0],
                                                            [                             0,                              0],
                                                            [1.0/self.sim_params.tau_player,                              0],
                                                            [                             0, 1.0/self.sim_params.tau_player]]),
                                                  np.eye(4),
                                                  np.zeros((4, 2)))
        self.controller_params.D = 3.0
        self.controller_params.k_lambda = 10.0
        self.controller_params.d_safe = 1.75
        self.controller_params.k_tang = 5.0
        self.controller_params.d_band = 0.5
        self.controller_params.Omega_n_u = np.array([[10.0,  0.0],
                                                     [ 0.0, 10.0]])
        self.controller_params.Omega_N_max = np.array([[10.0,  0.0,  0.0,  0.0],
                                                       [ 0.0, 10.0,  0.0,  0.0],
                                                       [ 0.0,  0.0, 20.0,  0.0],
                                                       [ 0.0,  0.0,  0.0, 20.0]])
        self.controller_params.Omega_N_min = np.array([[1.0, 0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 1.0]])

        # Declare strategy parameters
        self.strategy_params = StrategyParams()
        self.strategy_params.v_thresh = self.sim_params.arena_limits_x / 10.0
        self.strategy_params.d_goalie_offense = self.sim_params.arena_limits_x / 2.0
        self.strategy_params.d_goalie_defense = self.sim_params.arena_limits_x / 4.0
        self.strategy_params.v_hit = 4.0 # 6.0

        # Declare players
        self.attacker_player = DMPCPlayer(self.sim_params, self.controller_params, self.strategy_params,
                                          field, DMPCPlayer.ATTACKER)
        self.defender_player = DMPCPlayer(self.sim_params, self.controller_params, self.strategy_params,
                                          field, DMPCPlayer.DEFENDER)

    def execute(self, sim_state):
        # Determine the appropriate play
        puck_pos_x = sim_state.get_puck_pos()[0]
        puck_vel_x = sim_state.get_puck_vel()[0]
        if self.field * puck_pos_x > self.sim_params.arena_limits_x - self.strategy_params.d_goalie_defense:
            play = DMPCPlayer.DEFENSE
        elif (self.field * puck_pos_x > self.sim_params.arena_limits_x - self.strategy_params.d_goalie_offense) and (self.field * puck_vel_x > self.strategy_params.v_thresh):
            play = DMPCPlayer.DEFENSE
        else:
            play = DMPCPlayer.OFFENSE

        # Let the players play
        u_attacker = self.attacker_player.get_action(play, sim_state)
        u_defender = self.defender_player.get_action(play, sim_state)

        return u_attacker, u_defender
