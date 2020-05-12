import numpy as np
import copy

class SimState():
    def __init__(self, state):
        self.TAS = 0
        self.TBS = 1
        self.PK = 2
        self.A1 = 6
        self.A2 = 10
        self.B1 = 14
        self.B2 = 18
        self.state = copy.deepcopy(state)

    def get_player_pos(self, team, player_id):
        return copy.deepcopy(self.get_player_state(team, player_id)[0:2])

    def get_player_vel(self, team, player_id):
        return copy.deepcopy(self.get_player_state(team, player_id)[2:4])

    def get_player_state(self, team, player_id):
        if team == "A" and player_id == 1:
            return self.state[self.A1:(self.A1+4)]
        elif team == "A" and player_id == 2:
            return self.state[self.A2:(self.A2+4)]
        elif team == "B" and player_id ==1:
            return self.state[self.B1:(self.B1+4)]
        elif team == "B" and player_id == 2:
            return self.state[self.B2:(self.B2+4)]
        else:
            print("Team or player not recognized! team = A or B, player_id = 1 or 2")

    def get_puck_pos(self):
        return copy.deepcopy(self.get_puck_state()[0:2])

    def get_puck_vel(self):
        return copy.deepcopy(self.get_puck_state()[2:4])

    def get_puck_state(self):
        return copy.deepcopy(self.state[self.PK:(self.PK+4)])

    def transpose(self):
        return copy.deepcopy(self.state.transpose())
