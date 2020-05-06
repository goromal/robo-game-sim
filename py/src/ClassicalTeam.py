import numpy as np
from src.ClassicalPlayer import ClassicalPlayer

class ClassicalTeam:
    def __init__(self, params, field, team, state):
        self.params = params
        self.field = field
        self.team = team
        self.goalie = ClassicalPlayer(params, field, self.team, 1, state)
        self.player = ClassicalPlayer(params, field, self.team, 2, state)
        self.curr_play =  "defense" # one of ["defense", "offense",]
        self.kick_velocity = 2
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
        # Defense if the ball is in our field and the ball is moving towards our goal
        if self.field * state.get_puck_pos()[0] >= 0 and self.field * state.get_puck_vel()[0] >= 0:
            return "defense"
        else:
            return "offense"


    def execute(self, state):
        """Execute defense of offense strategies."""
        opp_pos1 = state.get_player_pos(self.get_adversary_team(), 1)
        opp_pos2 = state.get_player_pos(self.get_adversary_team(), 2)
        goalie_pos = state.get_player_pos(self.get_adversary_team(), 1)
        player_pos = state.get_player_pos(self.get_adversary_team(), 2)
        puck_pos = state.get_puck_pos()

        goalie_dist_from_puck = np.linalg.norm(goalie_pos - puck_pos)
        player_dist_from_puck = np.linalg.norm(player_pos - puck_pos)
        opp1_dist_from_puck = np.linalg.norm(opp_pos1 - puck_pos)
        opp2_dist_from_puck = np.linalg.norm(opp_pos2 - puck_pos)

        if self.curr_play == "offense":
            self.player.simple_kick(state,  self.kick_velocity)

            if self.field * state.get_puck_pos()[0] >= 0:
                self.goalie.simple_kick(state, self.kick_velocity)
            else:
                self.goalie.defend(state)

        elif self.curr_play == "defense":
            # if opponents are not too close, goalie kicks away the puck
            if goalie_dist_from_puck < opp1_dist_from_puck and goalie_dist_from_puck < opp2_dist_from_puck:
                self.goalie.defend_kick(state, self.kick_velocity)
            else:
                self.goalie.defend(state)

            # player tries to intercept the ball
            self.player.defend_kick(state,  self.kick_velocity)

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
