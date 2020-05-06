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
        self.kick_velocity = 4.5

        self.bounce_kick_planned = False

        # Main team logic. Takes sim_state and returns vel cmds.
    def run(self, state):

        ### Add state machine here
        next_play = self.evaluateGame(state)

        if next_play != self.curr_play:
            self.clean_up()
            self.curr_play = next_play
            self.set_up()
            self.execute(state)
        else:
            self.execute(state)

        ### # Call state transitions for goalie
        #if self.goalie.is_idle():
        #self.goalie.defend(state)

        ## # Plan an open loop kick if not doing it already!
        #if self.player.is_idle():
        #self.player.simple_kick_avoiding_obs(state, 5.0)

        # Do not change below here
        vel_goalie, _ = self.goalie.get_control()
        vel_player, _ = self.player.get_control()
        return vel_goalie, vel_player

    def evaluateGame(self, state):
        """Determine what play to use"""
        # TODO: offense if the ball is in our field
        #       defense if at least one of the players is in our field
        #       one player should actively intercept the ball when opponents are attacking

        return "defense"

    def execute(self, state):
        """Execute current play"""
        if self.curr_play == "idle":
            self.goalie.idle()
            self.player.idle()

        elif self.curr_play == "offense":
            if self.player.is_idle():
                time = 2.0
                vel = 5.0
                # self.player.timed_kick_avoiding_obs(state, vel, time)
                self.player.simple_kick(state,  self.kick_velocity) #
            self.goalie.defend(state)

        elif self.curr_play == "defense":
            self.player.defend(state)
            self.goalie.defend(state)

            # #################################
            # # Jeremy: Test contact optimizer
            # which_wall = "down"
            # if self.team == "B":
            #     self.goalie.defend(state)
            #     if not self.bounce_kick_planned:
            #         self.player.bounce_kick(state, which_wall)
            #         self.bounce_kick_planned = True
            # else:
            #     # self.goalie.idle()
            #     # self.player.idle()
            #     self.player.defend(state)
            #     self.goalie.defend(state)
            # #################################

    def clean_up(self):
        """Clean up for current play"""
        self.player.idle()
        self.goalie.idle()

    def set_up(self):
        """Set up for current play"""
