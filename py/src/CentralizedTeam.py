import numpy as np
from src.CentralizedPlayers import CentralizedPlayers

class ClassicalTeam:
    def __init__(self, sim_params, field, team, state):
        self.sim_params = sim_params
        self.field = field
        self.team = team
        self.players = CentralizedPlayers(sim_params)

        self.curr_play = "idle" # one of ["idle", "offense", "defense"]

    # Main team logic. Takes sim_state and returns vel cmds.
    def run(self, state):
        
        vel_player1, vel_player2 = self.players.attack(state)

        return vel_player1, vel_player2
