import numpy as np
from src.ClassicalPlayer import ClassicalPlayer

class ClassicalTeam:
    def __init__(self, params, field, team):
        self.params = params
        self.field = field
        self.team = team
        self.goalie = ClassicalPlayer(params, field, self.team, 1)
        self.player = ClassicalPlayer(params, field, self.team, 2)
        self.curr_play = "defense" # one of ["defense", "offense"]
        self.kick_velocity = 4
        # self.bounce_kick_planned = False

    def run(self, state):
        """Simple strategy based on state machine."""
        next_play = self.evaluateGame(state)

        if next_play != self.curr_play:
            self.clean_up()
            self.curr_play = next_play
            self.execute(state)
        else:
            self.execute(state)

        # Do not change below here
        vel_goalie, _ = self.goalie.get_control()
        vel_player, _ = self.player.get_control()

        return vel_goalie, vel_player

    def evaluateGame(self, state):
        # Defense if the ball is in our field, and the ball is moving towards our goal
        if self.field * state.get_puck_pos()[0] >= self.params.arena_limits_x/4.0 and self.field * state.get_puck_vel()[0] >= 0:
            return "defense"
        else:
            return "offense"

    def execute(self, state):
        """Execute defense of offense strategies. Note that the classical strategy is open-loop, i.e., each planned
        trajectory has to be completed, before planning the next one."""

        if self.curr_play == "offense":
            # player kicks the ball towards the goal
            if self.player.is_idle():
                self.player.simple_kick(state, self.kick_velocity)

            # goalie defends if the ball is in the home field, else attacks
            if self.goalie.is_idle():
                if self.field * state.get_puck_pos()[0] >= 0:
                    self.goalie.defend(state)
                else:
                    self.goalie.simple_kick(state, self.kick_velocity)

        elif self.curr_play == "defense":
            # player tries to hit the ball much harder to deflect the ball
            if self.player.is_idle():
                # self.player.simple_kick(state,  2 * self.kick_velocity)
                self.player.defend_kick(state,  1.5 * self.kick_velocity)

            # if opponents are not too close, goalie kicks away the puck, else defends
            if self.goalie.is_idle():
                opp_pos1 = state.get_player_pos(self.get_adversary_team(), 1)
                opp_pos2 = state.get_player_pos(self.get_adversary_team(), 2)
                goalie_pos = state.get_player_pos(self.get_adversary_team(), 1)
                puck_pos = state.get_puck_pos()

                goalie_dist_from_puck = np.linalg.norm(goalie_pos - puck_pos)
                opp1_dist_from_puck = np.linalg.norm(opp_pos1 - puck_pos)
                opp2_dist_from_puck = np.linalg.norm(opp_pos2 - puck_pos)

                if goalie_dist_from_puck < opp1_dist_from_puck and goalie_dist_from_puck < opp2_dist_from_puck:
                    self.goalie.defend_kick(state, self.kick_velocity)
                else:
                    self.goalie.defend(state)

    def clean_up(self):
        """Clean up old trajectories."""
        self.player.idle()
        self.goalie.idle()

    def get_adversary_team(self):
        """Returns the team adversary to the player's team"""
        if self.team == "A":
            return "B"
        elif self.team == "B":
            return "A"
        else:
            raise Exception("Team not recognized! Team can either be \"A\" or \"B\"")
