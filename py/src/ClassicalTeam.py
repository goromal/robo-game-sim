import numpy as np
from src.ClassicalPlayer import ClassicalPlayer

class ClassicalTeam:
    def __init__(self, params, field, team, state):
        self.params = params
        self.field = field
        self.team = team
        self.goolie = ClassicalPlayer(params, field, self.team, 1, state)
        self.player = ClassicalPlayer(params, field, self.team, 2, state)

    # Main team logic. Takes sim_state and returns vel cmds.
    def run(self, state):

        ## Add state machine here
        # Call state transitions for goolie
        self.goolie.idle() # this will tell the goolie to do nothing

        # Plan an open loop kick if not doing it already!
        if self.player.is_idle(): 
            self.player.timed_kick(state, 1.0)
        
        # Do not change below here
        vel_goolie, _ = self.goolie.get_control()
        vel_player, _ = self.player.get_control()
        return vel_goolie, vel_player
