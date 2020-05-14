import numpy as np
from src.JointPuckPlayerMPC import JointPuckPlayerMPC

class JointCentralizedTeam:
    def __init__(self, sim_params, field, team):
        self.sim_params = sim_params
        self.field = field
        self.team = team
        #self.players = BaselineCentralizedPlayers(sim_params, self.field, self.team)
        #self.players  = JointPuckPlayerMPC(sim_params)
        self.joint_puck_player_controller = JointPuckPlayerMPC(sim_params)

        # to run openloop
        self.u = np.zeros((2,1))
        self.T = -1
        self.t = 0

    # Main team logic. Takes sim_state and returns vel cmds.
    def run(self, state):
        x_puck = state.get_puck_state()
        p_goal  = self.get_adversary_goal_pos()
        x_p1 = state.get_player_state(self.team, 1)
        vel_player1, vel_player2 = self.joint_puck_player_controller.compute_control(x_p1, x_puck, p_goal)
        print("commanding:", vel_player1, vel_player2)
        return vel_player1, vel_player2

    def run_openloop(self, state): 
        while self.t < self.T-1:
            self.t = self.t + 1
            print("Erorr os hsere; ", self.u1[self.t])
            return self.u1[self.t, :], np.zeros(2)

        # compute new open-loop trajectory
        x_puck = state.get_puck_state()
        p_goal  = self.get_adversary_goal_pos()
        x_p1 = state.get_player_state(self.team, 1)
        self.u1, _ = self.joint_puck_player_controller.compute_control(x_p1, x_puck, p_goal)
        self.T = len(self.u1[:,1])
        print("T: ", self.T)
        print("u1: ", self.u1)
        self.t = 0
        return self.u1[self.t, :], np.zeros(2)

    def get_adversary_goal_pos(self):
        """Where the goal should be kicked."""
        if self.field > 0:
            return np.array([-self.sim_params.arena_limits_x/2.0, 0.0])
        else :
            return np.array([self.sim_params.arena_limits_x/2.0, 0.0])
