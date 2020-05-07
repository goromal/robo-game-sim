
import numpy as np
from src.CentralizedMPC import CentralizedMPC, LinearSystem

class MpcParams():
    def __init__(self, params):
        self.params = params

        self.A_player = np.eye(4) + self.params.dt*np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_player, 0], [0, 0, 0, -1.0/self.params.tau_player]])
        self.B_player = self.params.dt*np.array([[0, 0], [0, 0], [1.0/self.params.tau_player, 0], [0, 1.0/self.params.tau_player]])
        self.C_player = np.eye(4)
        self.D_player = np.zeros((4,2))
        self.sys_player = LinearSystem(self.A_player, self.B_player, self.C_player, self.D_player, self.params.dt)

        self.A_puck = np.eye(4) + self.params.dt*np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_puck, 0], [0, 0, 0, -1.0/self.params.tau_puck]])
        self.B_puck = np.zeros((4, 2))
        self.C_puck = np.eye(4)
        self.D_puck = np.zeros((4,2))
        self.sys_puck = LinearSystem(self.A_puck, self.B_puck, self.C_puck, self.D_puck, self.params.dt)

        self.A_player_c = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_player, 0], [0, 0, 0, -1.0/self.params.tau_player]])
        self.B_player_c = np.array([[0, 0], [0, 0], [1.0/self.params.tau_player, 0], [0, 1.0/self.params.tau_player]])
        self.C_player_c = np.eye(4)
        self.D_player_c = np.zeros((4,2))
        self.sys_c = LinearSystem(self.A_player_c, self.B_player_c, self.C_player, self.D_player_c)

        self.A_players_c = np.zeros((8,8))
        self.A_players_c[0:4, 0:4] = self.A_player_c
        self.A_players_c[4:8, 4:8] = self.A_player_c
        self.B_players_c = np.zeros((8,4))
        self.B_players_c[0:4,0:2] = self.B_player_c
        self.B_players_c[4:8,2:4] = self.B_player_c
        self.sys_two_players_c = LinearSystem(self.A_players_c, self.B_players_c, np.eye(8), np.zeros((8,4)))

        self.Q_puck = np.eye(4) # TODO: penalize velocity differently
        self.N = 20
        self.minT = self.params.dt/(self.N + 1)
        self.maxT = 4.0*self.params.dt
        self.Omega_N_max = np.array([[10.0,  0.0,  0.0,  0.0],[ 0.0, 10.0,  0.0,  0.0],[ 0.0,  0.0, 20.0,  0.0],[ 0.0,  0.0,  0.0, 20.0]])
        #self.Q_puck[2:4, 2:4] = np.zeros((2,2))


class CentralizedPlayers():

    def __init__(self, params, field, team):
        # player parameters
        self.sim_params = params
        self.mpc_params = MpcParams(params)
        self.field = field
        self.team = team
        self.player_1_id = 1 # Contains and controls both the players
        self.player_2_id = 2

        # controller
        self.controller = CentralizedMPC(self.sim_params, self.mpc_params)

    def attack(self, state):
        """compute control action for player1 and 2 to send the puck in the goal"""
        x0_p1 = state.get_player_state(self.team, self.player_1_id)
        x0_p2 = state.get_player_state(self.team, self.player_2_id)
        xf_p1 = np.zeros(4)
        xf_p2 = np.zeros(4)
        x_puck = state.get_puck_state()
        x_goal = self.get_adversary_goal_pos()
        obstacles = self.get_pos_of_other_players(state)
        converged, cmd1, cmd2 = self.controller.compute_control(x0_p1, x0_p2, xf_p1, xf_p2, x_puck, obstacles)
        return cmd1, cmd2

    # Where the ball should be kicked
    # Note: code duplicated from classical player :(
    def get_adversary_goal_pos(self):
        """returns the poosition of the goal of the adversary team"""
        if self.field > 0:
            return np.array([-self.sim_params.arena_limits_x/2.0, 0.0])
        else :
            return np.array([self.sim_params.arena_limits_x/2.0, 0.0])

    def get_pos_of_other_players(self, state):
        positions = list()
        positions.append(state.get_player_pos(self.get_adversary_team(), 1))
        positions.append(state.get_player_pos(self.get_adversary_team(), 2))
        return positions

    # Note: code duplicated from classical player :(
    def get_adversary_team(self):
        """Returns the team adversary to the player's team"""
        if self.team == "A":
            return "B"
        elif self.team == "B":
            return "A"
        else:
            raise Exception("Team not recognized! Team can either be \"A\" or \"B\"")