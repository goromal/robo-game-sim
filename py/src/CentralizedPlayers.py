
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

# Code from Andrew
class BaselineCentralizedPlayers(object):
    # play types
    OFFENSE = 0
    DEFENSE = 1
    # player types

    def __init__(self, sim_params, field, player_id):
        # player parameters
        self.params = sim_params
        self.v_hit = 5.0
        # field tells which side of the arena the player defends -1 left, +1 right
        self.home_pos =  field * np.array([self.params.arena_limits_x/2.0, 0.0])
        self.goal_pos = -field * np.array([self.params.arena_limits_x/2.0, 0.0])
        if field < 0: self.this_team = "A"
        else: self.this_team = "B"
        self.player_id = player_id # 1 or 2; 1 plays more offense, 2 plays more defense
        self.field = field
        self.attacker_id = 1 # Contains and controls both the players
        self.defender_id = 2
        self.mpc_params = MpcParams(self.params)

        # controller
        self.controller = CentralizedMPC(self.params, self.mpc_params)

    def get_action(self, play, state):
        x_des = np.zeros((4,1))
        puck_pos = state.get_puck_pos()

        if   play == BaselineCentralizedPlayers.OFFENSE:
            # Get to puck wherever it is and hit it towards the goal!
            hit_dir = self.goal_pos - puck_pos
            hit_dir = self.v_hit * hit_dir / np.linalg.norm(hit_dir)
            x_des_attack = np.array([puck_pos[0], puck_pos[1], hit_dir[0], hit_dir[1]])
            # If puck is within range, hit towards goal! Else defend.
            if self.field * puck_pos[0] > 0:
                def_pos = self.home_pos + (puck_pos - self.home_pos) / 2.0
                x_des_defend = np.array([def_pos[0], def_pos[1], 0., 0.])
            else:
                hit_dir = self.goal_pos - puck_pos
                hit_dir = self.v_hit * hit_dir / np.linalg.norm(hit_dir)
                x_des_defend = np.array([puck_pos[0], puck_pos[1], hit_dir[0], hit_dir[1]])

        elif play == BaselineCentralizedPlayers.DEFENSE:
            # Deflect puck trajectory in any way possible to return to offense
            hit_dir = self.goal_pos - puck_pos
            hit_dir = 2 * self.v_hit * hit_dir / np.linalg.norm(hit_dir)
            x_des_attack = np.array([puck_pos[0], puck_pos[1], hit_dir[0], hit_dir[1]])
            # Get between the puck and the home goal
            def_pos = self.home_pos + (puck_pos - self.home_pos) / 2.0
            x_des_defend = np.array([def_pos[0], def_pos[1], 0., 0.])
        
        x0_attacker = state.get_player_state(self.this_team, self.attacker_id)
        x0_defender = state.get_player_state(self.this_team, self.defender_id)
        obstacles = self.get_pos_of_other_players(state)
        _, cmd1, cmd2 = self.controller.compute_control(x0_attacker, x0_defender, x_des_attack, x_des_defend, obstacles)
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
        if self.this_team == "A":
            return "B"
        elif self.this_team == "B":
            return "A"
        else:
            raise Exception("Team not recognized! Team can either be \"A\" or \"B\"")