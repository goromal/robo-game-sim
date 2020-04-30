import numpy as np
import matplotlib.pyplot as plt

# pydrake imports
from pydrake.all import eq, MathematicalProgram, Solve, Variable, LinearSystem, DirectTranscription, DirectCollocation, PiecewisePolynomial

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
        self.continuous_sys = LinearSystem(self.A_c, self.B_c, self.C, self.D)
 
    # given initial state, final state, final time (Jeremy's)
    def intercepting_traj(self, p0, v0, pf, vf, T):
        x0 = np.concatenate((p0, v0), axis=0)
        xf = np.concatenate((pf, vf), axis=0)
        prog = DirectTranscription(self.sys, self.sys.CreateDefaultContext(), int(T/self.params.dt))
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        prog.AddBoundingBoxConstraint(xf, xf, prog.final_state())

        # input limits
        self.add_input_limits(prog)

        # remain inside arena
        self.add_arena_limits(prog)

        # cost
        prog.AddRunningCost(prog.input()[0]*prog.input()[0] + prog.input()[1]*prog.input()[1])

        # solve optimization
        result = Solve(prog)
        u_sol = prog.ReconstructInputTrajectory(result)
        if not result.is_success():
            print("Intercepting trajectory: optimization failed")

        u_values = u_sol.vector_values(u_sol.get_segment_times())
        return result.is_success(), u_values
    
    # TODO: minimum time trajectory
    # Reach a certain position in minimum time, regardless of anything else.
    def min_time_traj_transcription(self, p0, v0, pf, vf, xlim=None, ylim=None):
        """generate minimum time trajectory while avoiding obs"""
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

        u_values = u_sol.vector_values(u_sol.get_segment_times())
        return result.is_success(), u_values

    def min_time_traj(self, p0, v0, pf, vf, xlim=None, ylim=None):
        return self.min_time_traj_transcription(p0, v0, pf, vf)
        #return self.min_time_traj_dir_col(p0, v0, pf, vf)

    def min_time_traj_dir_col(self, p0, v0, pf, vf, xlim=None, ylim=None):
        """generate minimum time trajectory while avoiding obs"""
        T = 2
        N = int(T/self.params.dt)
        N = 21
        x0 = np.concatenate((p0, v0), axis=0)
        xf = np.concatenate((pf, vf), axis=0)
        prog = DirectCollocation(self.continuous_sys, self.continuous_sys.CreateDefaultContext(), N, minimum_timestep=0.05, maximum_timestep=0.2)
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())     # initial states
        prog.AddBoundingBoxConstraint(xf, xf, prog.final_state())
        
        prog.AddEqualTimeIntervalsConstraints()   

        self.add_input_limits(prog)
        self.add_arena_limits(prog)

        #R = 10  # Cost on input "effort".
        #u = prog.input()
        #prog.AddRunningCost(R * u[0]**2)

        initial_x_trajectory = PiecewisePolynomial.FirstOrderHold([0., 4.], np.column_stack((x0, xf)))  # yapf: disable
        prog.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

        prog.AddFinalCost(prog.time())

        result = Solve(prog)
        if not result.is_success():
            print("Minimum time trajectory: optimization failed")

        u_trajectory = prog.ReconstructInputTrajectory(result)
        T = u_trajectory.end_time() - u_trajectory.start_time()
        N_sol = int(T/self.params.dt)
        times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), 100)
        u_lookup = np.vectorize(u_trajectory.value)
        u_values = u_lookup(times)
        #return result.is_success(), u_values

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
    