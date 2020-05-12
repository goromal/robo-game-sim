import numpy as np
import math
from pydrake.all import eq, DirectCollocation, SnoptSolver

class DMPC(object):

    ATTACKER = 1  # plays more offense on average
    DEFENDER = 2  # plays more defense on average

    def __init__(self, sim_params, mpc_params):
        self.sim_params = sim_params
        self.mpc_params = mpc_params
        self.prev_u = None
        self.prev_x = None

    def compute_control(self, x_des, sim_state, team_name, player_id):
        prog = DirectCollocation(self.mpc_params.sys, self.mpc_params.sys.CreateDefaultContext(), self.mpc_params.N,
                                 minimum_timestep=self.mpc_params.minT, maximum_timestep=self.mpc_params.maxT)

        pos0 = sim_state.get_player_pos(team_name, player_id)
        vel0 = sim_state.get_player_vel(team_name, player_id)
        x0 = np.concatenate((pos0, vel0), axis=0)
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        prog.AddQuadraticErrorCost(Q=self.mpc_params.Omega_N_max, x_desired=x_des, vars=prog.final_state())


        obstacle_positions = self.get_obstacle_positions(sim_state, team_name, player_id)
        for obs_pos in obstacle_positions:
            for n in range(self.mpc_params.N):
                x = prog.state()
                prog.AddConstraintToAllKnotPoints((x[0:2]-obs_pos).dot(x[0:2]-obs_pos) >= (2.0*self.sim_params.player_radius)**2)

        prog.AddEqualTimeIntervalsConstraints()

        self.add_input_limits(prog)
        self.add_arena_limits(prog)

        prog.AddFinalCost(prog.time())

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

    def get_obstacle_positions(self, sim_state, team_name, player_id):
        obstacle_positions = list()
        all_agents = [("A", DMPC.ATTACKER), ("A", DMPC.DEFENDER), ("B", DMPC.ATTACKER), ("B", DMPC.DEFENDER)]
        for agent in all_agents:
            agent_team = agent[0]
            agent_id   = agent[1]
            if not (agent_team == team_name and agent_id == player_id):
                obstacle_positions.append(sim_state.get_player_pos(agent_team, agent_id))
        return obstacle_positions

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

    # # Convex obstacle avoidance MPC algorithm <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #
    # Omega_N = self.mpc_params.Omega_N_max
    #
    # flag_minimum_time = True
    #
    # f_k = list()
    # f_k.append(np.zeros(2))
    # total_tangent_force = np.zeros(2)
    #
    # for n in range(1, self.mpc_params.N + 1):
    #     p_k_nm1 = sim_state.get_player_pos(team_name, player_id) + (n - 1) * self.dt * sim_state.get_player_vel(team_name, player_id)
    #     obstacle_positions = self.get_obstacle_positions(sim_state, team_name, player_id, n, self.dt)
    #     N_k = [obs_pos for obs_pos in obstacle_positions if np.linalg.norm(obs_pos - p_k_nm1) < self.mpc_params.D]
    #
    #     f_k_n = np.zeros(2)
    #
    #     for p_j_n in N_k:
    #         d_j_n = np.linalg.norm(p_k_nm1 - p_j_n)
    #         alpha_j_n = (p_j_n - p_k_nm1) / np.linalg.norm(p_j_n - p_k_nm1)
    #
    #         F_rep_kj_n = np.zeros(2)
    #         if d_j_n < self.mpc_params.d_safe:
    #             F_rep_kj_n = -(self.mpc_params.k_lambda * math.pi * math.sin(math.pi * d_j_n / self.mpc_params.d_safe)) / (2 * self.mpc_params.d_safe * math.sin(math.pi * d_j_n / (2 * self.mpc_params.d_safe))**4) * alpha_j_n
    #
    #         # print(f_k_n.shape, F_rep_kj_n.shape)
    #         f_k_n = f_k_n + F_rep_kj_n
    #
    #         if n == 1:
    #             if d_j_n > self.mpc_params.d_safe and d_j_n < self.mpc_params.d_safe + self.mpc_params.d_band:
    #                 flag_minimum_time = False
    #                 Omega_N_interp = self.mpc_params.Omega_N_min + (d_j_n - self.mpc_params.d_safe) / self.mpc_params.d_band * (self.mpc_params.Omega_N_max - self.mpc_params.Omega_N_min)
    #                 if Omega_N_interp[0,0] < Omega_N[0,0]:
    #                     Omega_N = Omega_N_interp
    #                 delJ_k = (d_j_n - self.mpc_params.d_safe) / self.mpc_params.d_band
    #                 alpha_hat = np.array([-alpha_j_n[1], alpha_j_n[0]])
    #                 F_tang_kj_1 = self.mpc_params.k_tang * delJ_k * alpha_hat
    #             else:
    #                 F_tang_kj_1 = np.zeros(2)
    #             total_tangent_force = total_tangent_force + F_tang_kj_1
    #
    #     f_k_n = f_k_n + total_tangent_force
    #
    #     # f_k.append(f_k_n)
    #     f_k.append(np.zeros(2)) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
