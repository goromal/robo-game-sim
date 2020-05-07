import numpy as np
import math
import copy
from pydrake.all import eq, DirectCollocation, DirectTranscription, SnoptSolver, Solve, MathematicalProgram, Variable, LinearSystem, GetInfeasibleConstraints

class CentralizedMPC():

    def __init__(self, sim_params, mpc_params):
        self.sim_params = sim_params
        self.mpc_params = mpc_params
        self.prev_xp = None

        # To generate intial guess for one player
        self.prev_u = None
        self.prev_x = None

    def compute_control(self, x0_p1, x0_p2, xf_p1, xf_p2, x_puck, obstacles):
        """This is basically the single-agent MPC algorithm"""
        prog = DirectCollocation(self.mpc_params.sys_two_players_c, self.mpc_params.sys_two_players_c.CreateDefaultContext(), self.mpc_params.N+1,
                    minimum_timestep=self.mpc_params.minT, maximum_timestep=self.mpc_params.maxT)
        x0 = np.concatenate((x0_p1, x0_p2), axis=0)
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        x_des = np.concatenate((xf_p1, xf_p2))
        Q = np.zeros((8,8))
        Q[0:4,0:4] = self.mpc_params.Omega_N_max
        Q[4:8,4:8] = self.mpc_params.Omega_N_max
        prog.AddQuadraticErrorCost(Q, x_desired=x_des, vars=prog.final_state())

        prog.AddEqualTimeIntervalsConstraints()

        for obs_pos in obstacles: # both players should avoid the other players
            for n in range(self.mpc_params.N):
                x = prog.state()
                prog.AddConstraintToAllKnotPoints((x[0:2]-obs_pos).dot(x[0:2]-obs_pos) >= (2.0*self.sim_params.player_radius)**2)
                prog.AddConstraintToAllKnotPoints((x[4:6]-obs_pos).dot(x[4:6]-obs_pos) >= (2.0*self.sim_params.player_radius)**2)

        # players should avoid each other
        prog.AddConstraintToAllKnotPoints((x[0:2]-x[4:6]).dot(x[0:2]-x[4:6]) >= (2.0*self.sim_params.player_radius)**2)
        
        # input constraints
        for i in range(4):
            prog.AddConstraintToAllKnotPoints(prog.input()[i] <=  self.sim_params.input_limit)
            prog.AddConstraintToAllKnotPoints(prog.input()[i] >= -self.sim_params.input_limit)

        r = self.sim_params.player_radius
        prog.AddConstraintToAllKnotPoints(prog.state()[0] + r <=  self.sim_params.arena_limits_x / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[0] - r >= -self.sim_params.arena_limits_x / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[1] + r <=  self.sim_params.arena_limits_y / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[1] - r >= -self.sim_params.arena_limits_y / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[4] + r <=  self.sim_params.arena_limits_x / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[4] - r >= -self.sim_params.arena_limits_x / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[5] + r <=  self.sim_params.arena_limits_y / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[5] - r >= -self.sim_params.arena_limits_y / 2.0)

        prog.AddFinalCost(prog.time())

        if not self.prev_u is None and not self.prev_x is None:
            prog.SetInitialTrajectory(traj_init_u=self.prev_u, traj_init_x=self.prev_x)

        solver = SnoptSolver()
        result = solver.Solve(prog)

        u_traj = prog.ReconstructInputTrajectory(result)
        x_traj = prog.ReconstructStateTrajectory(result)

        self.prev_u = u_traj
        self.prev_x = x_traj

        u_vals = u_traj.vector_values(u_traj.get_segment_times())
        x_vals = x_traj.vector_values(x_traj.get_segment_times())

        return True, u_vals[0:2,0], u_vals[2:4,0]

    def add_input_limits(self, prog, u, N):
        for k in range(N):
            bound = np.array([self.sim_params.input_limit, self.sim_params.input_limit])
            prog.AddBoundingBoxConstraint(-bound, bound, u[k])

    def add_arena_limits(self, prog, state, N):
        r = self.sim_params.player_radius
        bound = np.array([self.sim_params.arena_limits_x / 2.0 + r, self.sim_params.arena_limits_y / 2.0 + r])
        for k in range(N+1):
            prog.AddBoundingBoxConstraint(-bound, bound, state[k, 0:2])

    def get_final_state_for_kick(self, p_goal, p_puck, kick_velocity):
        shoot_direction = self.get_shoot_direction(p_goal, p_puck)
        pf = p_puck #- shoot_direction*(self.sim_params.puck_radius + self.sim_params.player_radius)
        vf = kick_velocity*shoot_direction
        return pf, vf

    def get_shoot_direction(self, p_goal, p_puck):
        """Returns direction to kick to reach goal"""
        shoot_direction = p_goal - p_puck
        if np.linalg.norm(shoot_direction) > 1e-4:
            shoot_direction/=np.linalg.norm(shoot_direction)
        return shoot_direction