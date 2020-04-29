import numpy as np
import matplotlib.pyplot as plt

# pydrake imports
from pydrake.all import eq, MathematicalProgram, Solve, Variable, LinearSystem, DirectTranscription, GetInfeasibleConstraints

class ContactOptimizer:
    """"""
    def __init__(self, params):
        self.params = params

    def bounce_pass_wall(self, p0, duration=3, debug=True):
        """Solve for initial velocity for the puck to bounce the wall once and hit the goal."""
        # initialize program
        prog = MathematicalProgram()

        # Time periods before and after hitting the wall
        h = prog.NewContinuousVariables(2)

        # velocities of the ball at the start of each segment (after collision).
        vel_start = prog.NewContinuousVariables(2)
        vel_after = prog.NewContinuousVariables(2)

        # sum of the durations meet the requirement
        prog.AddConstraint(np.sum(h) <= duration)

        # force to move down-right at the beginning
        prog.AddBoundingBoxConstraint([0, -np.inf], [np.inf, -0], vel_start)

        # add dynamics constraints for the moment the ball hits the wall
        p_contact = self.next_p(p0, vel_start, h[0])
        v_contact = self.next_vel(vel_start, h[0])

        # keep the same x vel, but flip the y vel
        self.add_reset_map_constraint(prog, p_contact, v_contact, vel_after)

        # in the last segment, need to specify bounds for the final position and velocity of the ball
        p_end = self.next_p(p_contact, vel_after, h[1])
        v_end = self.next_vel(vel_after, h[1])
        self.add_goal_constraint(prog, p_end, v_end)


        # solve for the periods
        result = Solve(prog)
        if not result.is_success():
            if debug:
                infeasible = GetInfeasibleConstraints(prog, result)
                print("Infeasible constraints in ContactOptimizer:")
                for i in range(len(infeasible)):
                    print(infeasible[i])
            # return directly
            return
        else:
            if debug:
                print("vel_start:{}".format(result.GetSolution(vel_start)))
                print("vel_after: {}".format( result.GetSolution(vel_after)))
                print("h: {}".format( result.GetSolution(h)))
                p1 = self.next_p(p0, result.GetSolution(vel_start), result.GetSolution(h[0]))
                p2 = self.next_p(p1, result.GetSolution(vel_after), result.GetSolution(h[1]))
                print("p1:{}".format(p1))
                print("p2:{}".format(p2))
                v_end = self.next_vel(result.GetSolution(vel_after), result.GetSolution(h[1]))
                print("v_end:{}".format(v_end))

        # return whether successful, and the initial puck velocity
        return result.is_success(), result.GetSolution(vel_start)

    def add_reset_map_constraint(self, prog, p_contact, v_contact, vel_after):
        """Assume lower wall."""
        prog.AddConstraint(p_contact[1] == -self.params.arena_limits_y/2.0 + self.params.puck_radius)
        prog.AddConstraint(vel_after[0] == v_contact[0])
        prog.AddConstraint(vel_after[1] == -1*v_contact[1])


    def next_vel(self, v0, h):
        return v0 * np.exp(-h / self.params.tau_puck)

    def next_p(self, p0, v0, h):
        return p0 + self.params.tau_puck * v0 * (1 - np.exp(-h / self.params.tau_puck))

    def add_goal_constraint(self, prog, p_end, v_end):
        p_goal = np.array([self.params.arena_limits_x/2.0, 0])
        prog.AddConstraint(p_end[0] == p_goal[0])
        prog.AddConstraint(p_end[1] == p_goal[1])
        prog.AddConstraint(v_end[0] >= 0.1)
        prog.AddConstraint(v_end[1] >= 0.1)