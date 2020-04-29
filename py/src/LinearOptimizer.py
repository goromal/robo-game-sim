import numpy as np
import matplotlib.pyplot as plt

# pydrake imports
from pydrake.all import eq, MathematicalProgram, Solve, Variable, LinearSystem, DirectTranscription

class LinearOptimizer:
    def __init__(self, params):
        self.params = params
        self.A = np.eye(4) + self.params.dt*np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_player, 0], [0, 0, 0, -1.0/self.params.tau_player]])
        self.B = self.params.dt*np.array([[0, 0], [0, 0], [1.0/self.params.tau_player, 0], [0, 1.0/self.params.tau_player]])
        self.C = np.eye(4)
        self.D = np.zeros((4,2))
        self.sys = LinearSystem(self.A, self.B, self.C, self.D, self.params.dt)
 
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
    def min_time_traj(self, p0, v0, pf, vf, xlim=None, ylim=None):
        T = 1
        x0 = np.concatenate((p0, v0), axis=0)
        xf = np.concatenate((pf, vf), axis=0)
        prog = DirectTranscription(self.sys, self.sys.CreateDefaultContext(), int(T/self.params.dt))
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

    def min_time_bounce_kick_traj(self, p0, v0, p0_puck, v0_puck, v_puck_desired):
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

        u_values = u_sol.vector_values(u_sol.get_segment_times())
        return result.is_success(), u_values

    # TODO: account for moving puck. Right now puck is assumed to be static
    def add_final_state_constraint_elastic_collision(self, prog, p0_puck, v0_puck, v_puck_desired):
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
        # v_puck_after_collision = v2 - 2*m1/(m1+m2)*(v2-v1).dot(p2-p1)/(p2-p1).dot(p2-p1)*(p2-p1) # doesn't work for some reason
        v_puck_after_collision = v2 - 2*m1/(m1+m2)*(v2-v1).dot(p2-pf)/(p2-pf).dot(p2-pf)*(p2-pf)
        prog.AddConstraint(eq(v_puck_after_collision, v_puck_desired))

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
