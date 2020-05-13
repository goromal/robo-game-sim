import numpy as np
import matplotlib.pyplot as plt

# pydrake imports
from pydrake.all import eq, MathematicalProgram, Solve, Variable, LinearSystem, DirectTranscription, DirectCollocation, PiecewisePolynomial, SnoptSolver

class LinearOptimizer:
    def __init__(self, params):
        self.params = params
        self.A_c = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_player, 0], [0, 0, 0, -1.0/self.params.tau_player]])
        self.A = np.eye(4) + self.params.dt*np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_player, 0], [0, 0, 0, -1.0/self.params.tau_player]])
        self.B = self.params.dt*np.array([[0, 0], [0, 0], [1.0/self.params.tau_player, 0], [0, 1.0/self.params.tau_player]])
        self.B_c = np.array([[0, 0], [0, 0], [1.0/self.params.tau_player, 0], [0, 1.0/self.params.tau_player]])
        self.C = np.eye(4)
        self.D = np.zeros((4,2))
        self.sys = LinearSystem(self.A, self.B, self.C, self.D, self.params.dt)
        self.sys_c = LinearSystem(self.A_c, self.B_c, self.C, self.D)

    def intercepting_traj(self, p0, v0, pf, vf, T):
        """Trajectory planning given initial state, final state, and final time."""
        x0 = np.concatenate((p0, v0), axis=0)
        xf = np.concatenate((pf, vf), axis=0)
        prog = DirectTranscription(self.sys, self.sys.CreateDefaultContext(), int(T/self.params.dt))
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        prog.AddBoundingBoxConstraint(xf, xf, prog.final_state())

        self.add_input_limits(prog)
        self.add_arena_limits(prog)

        # cost
        prog.AddRunningCost(prog.input()[0]*prog.input()[0] + prog.input()[1]*prog.input()[1])

        # solve optimization
        result = Solve(prog)
        u_sol = prog.ReconstructInputTrajectory(result)
        if not result.is_success():
            print("Intercepting trajectory: optimization failed")
            return False, np.zeros((2, 1))

        u_values = u_sol.vector_values(u_sol.get_segment_times())
        return result.is_success(), u_values

    def min_time_traj_transcription(self, p0, v0, pf, vf, xlim=None, ylim=None):
        """Minimum time traj using directi transcription."""
        T = 2
        N = int(T/self.params.dt)
        x0 = np.concatenate((p0, v0), axis=0)
        xf = np.concatenate((pf, vf), axis=0)
        prog = DirectTranscription(self.sys, self.sys.CreateDefaultContext(), N)
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())     # initial states
        prog.AddBoundingBoxConstraint(xf, xf, prog.final_state())

        self.add_input_limits(prog)
        self.add_arena_limits(prog)

        prog.AddFinalCost(prog.time())

        result = Solve(prog)
        u_sol = prog.ReconstructInputTrajectory(result)
        if not result.is_success():
            print("Minimum time trajectory: optimization failed")
            return False, np.zeros((2, 1))

        u_values = u_sol.vector_values(u_sol.get_segment_times())
        return result.is_success(), u_values

    def min_time_bounce_kick_traj(self, p0, v0, p0_puck, v0_puck, v_puck_desired):
        """Use direct transcription to calculate player's trajectory for bounce kick. The elastic collision is enforced
        when robot reaches the desired position at specified time."""
        T = 1
        x0 = np.concatenate((p0, v0), axis=0)
        prog = DirectTranscription(self.sys, self.sys.CreateDefaultContext(), int(T/self.params.dt))
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        self.add_final_state_constraint_elastic_collision(prog, p0_puck, v0_puck, v_puck_desired)
        self.add_input_limits(prog)
        self.add_arena_limits(prog)
        prog.AddFinalCost(prog.time())

        result = Solve(prog)
        u_sol = prog.ReconstructInputTrajectory(result)
        if not result.is_success():
            print("Minimum time bounce kick trajectory: optimization failed")
            return False, np.zeros((2, 1))

        u_values = u_sol.vector_values(u_sol.get_segment_times())
        return result.is_success(), u_values

    def min_time_bounce_kick_traj_dir_col(self, p0, v0, p0_puck, v0_puck, v_puck_desired):
        """DO NOT USE. NOT WORKING.
        Minimum time trajectory + bounce kick off the wall."""
        N = 15
        minT = self.params.dt / N
        maxT = 5.0 / N
        x0 = np.concatenate((p0, v0), axis=0)
        prog = DirectCollocation(self.sys_c, self.sys_c.CreateDefaultContext(), num_time_samples=N,
                                 minimum_timestep=minT,
                                 maximum_timestep=maxT)
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        prog.AddEqualTimeIntervalsConstraints()
        self.add_final_state_constraint_elastic_collision(prog, p0_puck, v0_puck, v_puck_desired)
        self.add_input_limits(prog)
        self.add_arena_limits(prog)

        # prog.AddQuadraticErrorCost(Q=10.0*np.eye(4), x_desired=xf, vars=prog.final_state())
        pf = p0_puck - self.get_normalized_vector(v_puck_desired)*(self.params.puck_radius + self.params.player_radius)
        prog.AddQuadraticErrorCost(Q=10.0*np.eye(2), x_desired=pf, vars=prog.final_state()[:2])

        prog.AddFinalCost(prog.time())

        solver = SnoptSolver()
        result = solver.Solve(prog)
        if not result.is_success():
            print("Minimum time trajectory: optimization failed")
            return False, np.zeros((2, 1))

        u_trajectory = prog.ReconstructInputTrajectory(result)
        times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), (u_trajectory.end_time() - u_trajectory.start_time()) / self.params.dt )

        u_values = np.empty((2, len(times)))
        for i, t in enumerate(times):
            u_values[:, i] = u_trajectory.value(t).flatten()

        return result.is_success(), u_values

    def add_final_state_constraint_elastic_collision(self, prog, p0_puck, v0_puck, v_puck_desired):
        """Utility function for bounce kick from the wall. Probably not a linear constraint."""
        m1 = self.params.player_mass
        m2 = self.params.puck_mass
        p1 = prog.final_state()[:2] # var: player's final position
        p2 = p0_puck
        v1 = prog.final_state()[2:] # var: player's final velocity
        v2 = v0_puck

        # Final position constraint
        pf = p0_puck - self.get_normalized_vector(v_puck_desired)*(self.params.puck_radius + self.params.player_radius)
        prog.AddConstraint(eq(p1, pf))

        # Final velocity constraint
        v_puck_after_collision = v2 - 2*m1/(m1+m2)*(v2-v1).dot(p2-pf)/(p2-pf).dot(p2-pf)*(p2-pf)    # perhaps not a nonlinear constraint, since dot product of v1 is required
        prog.AddConstraint(eq(v_puck_after_collision, v_puck_desired))

    def min_time_traj(self, p0, v0, pf, vf):
        return self.min_time_traj_dir_col(p0, v0, pf, vf)

    def min_time_traj_dir_col(self, p0, v0, pf, vf):
        """generate minimum time trajectory while avoiding obs"""
        N = 15
        minT = self.params.dt / N
        maxT = 5.0 / N
        x0 = np.concatenate((p0, v0), axis=0)
        xf = np.concatenate((pf, vf), axis=0)

        prog = DirectCollocation(self.sys_c, self.sys_c.CreateDefaultContext(), num_time_samples=N,
                                 minimum_timestep=minT,
                                 maximum_timestep=maxT)
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        prog.AddEqualTimeIntervalsConstraints()
        self.add_input_limits(prog)
        self.add_arena_limits(prog)

        prog.AddQuadraticErrorCost(Q=10.0*np.eye(4), x_desired=xf, vars=prog.final_state())
        prog.AddFinalCost(prog.time())

        solver = SnoptSolver()
        result = solver.Solve(prog)
        if not result.is_success():
            print("Minimum time trajectory: optimization failed")
            return False, np.zeros((2, 1))

        # subsample trajectory accordingly
        u_trajectory = prog.ReconstructInputTrajectory(result)
        duration = u_trajectory.end_time() - u_trajectory.start_time()
        if duration > self.params.dt:
            times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), (u_trajectory.end_time() - u_trajectory.start_time()) / self.params.dt )
        else:
            times = np.array([0])

        u_values = np.empty((2, len(times)))
        for i, t in enumerate(times):
            u_values[:, i] = u_trajectory.value(t).flatten()

        return result.is_success(), u_values

    def add_input_limits(self, prog):
        prog.AddConstraintToAllKnotPoints(prog.input()[0] <= self.params.input_limit)
        prog.AddConstraintToAllKnotPoints(prog.input()[0] >= -self.params.input_limit)
        prog.AddConstraintToAllKnotPoints(prog.input()[1] <= self.params.input_limit)
        prog.AddConstraintToAllKnotPoints(prog.input()[1] >= -self.params.input_limit)

    def add_arena_limits(self, prog):
        r = self.params.player_radius
        prog.AddConstraintToAllKnotPoints(prog.state()[0] + r <= self.params.arena_limits_x/2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[0] - r >= -self.params.arena_limits_x/2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[1] + r <= self.params.arena_limits_y/2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[1] -r >= -self.params.arena_limits_y/2.0)

    def get_normalized_vector(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
