import numpy as np
from src.ClassicalPlayer import ClassicalPlayer

class ClassicalTeam:
    def __init__(self, params, field, team, state):
        self.params = params
        self.field = field
        self.team = team
        self.goolie = ClassicalPlayer(params, field, self.team, 1, state)
        self.player = ClassicalPlayer(params, field, self.team, 2, state)

        self.curr_play = "idle" # one of ["idle", "offense", "defense"]

    # Main team logic. Takes sim_state and returns vel cmds.
    def run(self, state):

        ## Add state machine here
        next_play = self.evaluateGame(state)

        if next_play != self.curr_play:
            self.clean_up()
            self.curr_play = next_play
            self.set_up()
            self.execute(state)
        else:
            self.execute(state)

        # # Call state transitions for goolie
        # self.goolie.idle() # this will tell the goolie to do nothing
        #
        # # Plan an open loop kick if not doing it already!
        # if self.player.is_idle():
        #     self.player.timed_kick(state, 1.0)
        
        # Do not change below here
        vel_goolie, _ = self.goolie.get_control()
        vel_player, _ = self.player.get_control()
        return vel_goolie, vel_player

    def evaluateGame(self, state):
        """Determine what play to use"""
        return "offense"

    def execute(self, state):
        """Execute current play"""
        if self.curr_play == "idle":
            self.goolie.idle()
            self.player.idle()

        elif self.curr_play == "offense":
            if self.player.is_idle():
                self.player.timed_kick(state, 1.0)
            self.goolie.defend(state)

        elif self.curr_play == "defense":
            self.player.defend(state)
            self.goolie.defend(state)


    def clean_up(self):
        """Clean up for current play"""
        self.player.idle()
        self.goolie.idle()

    def set_up(self):
        """Set up for current play"""
