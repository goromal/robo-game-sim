import numpy as np
import math
from pydrake.all import eq, DirectCollocation, SnoptSolver

class CentralizedMPC():

    def __init__(self, sim_params, mpc_params):
        self.sim_params = sim_params
        self.mpc_params = mpc_params
        self.prev_u = None
        self.prev_x = None

    def compute_control(self, x_p1, x_p2, x_puck, x_goal, obstacles):
        # TODO: Andrea: replace this code from Andrew
        #prog = DirectCollocation(self.mpc_params.sys, self.mpc_params.sys.CreateDefaultContext(), self.mpc_params.N,
        #                         minimum_timestep=self.mpc_params.minT, maximum_timestep=self.mpc_params.maxT)

        #pos0 = sim_state.get_player_pos(team_name, player_id)
        #vel0 = sim_state.get_player_vel(team_name, player_id)
        #x0 = np.concatenate((pos0, vel0), axis=0)
        #prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        #prog.AddQuadraticErrorCost(Q=self.mpc_params.Omega_N_max, x_desired=x_des, vars=prog.final_state())


        #obstacle_positions = self.get_obstacle_positions(sim_state, team_name, player_id)
        #for obs_pos in obstacle_positions:
        #    for n in range(self.mpc_params.N):
        #        x = prog.state()
        #        prog.AddConstraintToAllKnotPoints((x[0:2]-obs_pos).dot(x[0:2]-obs_pos) >= (2.0*self.sim_params.player_radius)**2)

        #prog.AddEqualTimeIntervalsConstraints()

        #self.add_input_limits(prog)
        #self.add_arena_limits(prog)

        #prog.AddFinalCost(prog.time())

        if not self.prev_u is None and not self.prev_x is None:
            prog.SetInitialTrajectory(traj_init_u=self.prev_u, traj_init_x=self.prev_x)

        solver = SnoptSolver()
        result = solver.Solve(prog)

        u_traj = prog.ReconstructInputTrajectory(result)
        x_traj = prog.ReconstructStateTrajectory(result)

        u_vals = u_traj.vector_values(u_traj.get_segment_times())

        self.prev_u = u_traj
        self.prev_x = x_traj

        return u_vals[:,0]

    def add_input_limits(self, prog):
        prog.AddConstraintToAllKnotPoints(prog.input()[0] <=  self.sim_params.input_limit)
        prog.AddConstraintToAllKnotPoints(prog.input()[0] >= -self.sim_params.input_limit)
        prog.AddConstraintToAllKnotPoints(prog.input()[1] <=  self.sim_params.input_limit)
        prog.AddConstraintToAllKnotPoints(prog.input()[1] >= -self.sim_params.input_limit)

    def add_arena_limits(self, prog):
        r = self.sim_params.player_radius
        prog.AddConstraintToAllKnotPoints(prog.state()[0] + r <=  self.sim_params.arena_limits_x / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[0] - r >= -self.sim_params.arena_limits_x / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[1] + r <=  self.sim_params.arena_limits_y / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[1] - r >= -self.sim_params.arena_limits_y / 2.0)