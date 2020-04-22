import numpy as np
from src.ClassicalPlayer import ClassicalPlayer

class ClassicalTeam:
    def __init__(self, params, field, team, state):
        self.params = params
        self.field = field
        self.team = team
        self.goalie = ClassicalPlayer(params, field, self.team, 1, state)
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

        # # Call state transitions for goalie
        # self.goalie.idle() # this will tell the goalie to do nothing
        #
        # # Plan an open loop kick if not doing it already!
        # if self.player.is_idle():
        #     self.player.timed_kick(state, 1.0)
        
        # Do not change below here
        vel_goalie, _ = self.goalie.get_control()
        vel_player, _ = self.player.get_control()
        return vel_goalie, vel_player

    def evaluateGame(self, state):
        """Determine what play to use"""
        return "offense"

    def execute(self, state):
        """Execute current play"""
        if self.curr_play == "idle":
            self.goalie.idle()
            self.player.idle()

        elif self.curr_play == "offense":
            if self.player.is_idle():
                self.player.timed_kick(state, 2.0)
                print("Timed kick towards puck which is at", state.get_puck_pos())
            self.goalie.defend(state)

        elif self.curr_play == "defense":
            self.player.defend(state)
            self.goalie.defend(state)


    def clean_up(self):
        """Clean up for current play"""
        self.player.idle()
        self.goalie.idle()

    def set_up(self):
        """Set up for current play"""
