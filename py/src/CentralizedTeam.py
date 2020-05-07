import numpy as np
from src.CentralizedPlayers import BaselineCentralizedPlayers

class CentralizedTeam:
    def __init__(self, sim_params, field, team, state):
        self.sim_params = sim_params
        self.field = field
        self.team = team
        self.players = BaselineCentralizedPlayers(sim_params, self.field, self.team)

        self.curr_play = "idle" # one of ["idle", "offense", "defense"]

    # Main team logic. Takes sim_state and returns vel cmds.
    def run(self, state):
        play = BaselineCentralizedPlayers.OFFENSE
        vel_player1, vel_player2 = self.players.get_action(BaselineCentralizedPlayers.OFFENSE, state)
        print("commanding:", vel_player1, vel_player2)
        return vel_player1, vel_player2
