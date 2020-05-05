
import numpy as np
from src.CentralizedMPC import CentralizedMPC

class MpcParams():
    self.todo = "todo"

class CentralizedPlayers():

    def __init__(self, sim_params, strategy_params, field, player_id):
        # player parameters
        self.sim_params = sim_params
        self.player_id = player_id # 1 or 2; 1 plays more offense, 2 plays more defense
        self.field = field
        self.player_1 = 1
        self.player_2 = 2

        # controller
        self.controller_params = MpcParams()
        self.controller = CentralizedMPC(sim_params, controller_params)

    def attack(self, state):
        """compute control action for player1 and 2 to send the puck in the goal"""
        x_p1 = state.get_player_state(self.player_1)
        x_p2 = state.get_player_state(self.player_2)
        x_puck = state.get_puck_state()
        x_goal = get_adversary_goal_pos()
        obstacles = get_pos_of_other_players(state)
        return self.controller.compute_control(x_p1, x_p2, x_puck, x_goal, obstacles)

    # Where the ball should be kicked
    # Note: code duplicated from classical player :(
    def get_adversary_goal_pos(self):
        """returns the poosition of the goal of the adversary team"""
        if self.field > 0:
            return np.array([-self.params.arena_limits_x/2.0, 0.0])
        else :
            return np.array([self.params.arena_limits_x/2.0, 0.0])

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