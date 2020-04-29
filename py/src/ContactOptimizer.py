import numpy as np
import matplotlib.pyplot as plt

# pydrake imports
from pydrake.all import eq, MathematicalProgram, Solve, Variable, LinearSystem, DirectTranscription, GetInfeasibleConstraints

class ContactOptimizer:
    """"""
    def __init__(self, params):
        self.params = params

    def bounce_pass_wall(self, p0, duration=2):
        """Solve for initial velocity for the puck to bounce the wall once and hit the goal."""
        number_of_bounces=1

        # initialize program
        prog = MathematicalProgram()

        ###########################
        # create decision variables
        ###########################
        # Time periods before and after hitting the wall
        h = prog.NewContinuousVariables(number_of_bounces+1)

        # velocities of the ball at the start of each segment (after collision).
        vels =  np.empty((2, number_of_bounces+1), dtype=Variable)
        for n in range(number_of_bounces+1):
            vels[:,n] = prog.NewContinuousVariables(2)

        # sum of the durations meet the requirement
        prog.AddConstraint(np.sum(h) <= duration)

        # force to move down-right at the beginning
        prog.AddBoundingBoxConstraint([0, -np.inf], [np.inf, 0], vels[:,0])

        # add dynamics constraints for the moment the ball hits the wall
        tau_puck = self.params.tau_puck
        p_contact = p0 + tau_puck*vels[:, 0]*(1-np.exp(-h[0]/tau_puck)) # close form solution for p
        v_contact = vels[:, 0]*np.exp(-h[0]/tau_puck)   # close form solution for v
        prog.AddConstraint(p_contact[1] == -self.params.arena_limits_y/2.0 + self.params.puck_radius)

        # keep the same x vel, but flip the y vel
        prog.AddConstraint(vels[0, 1] == v_contact[0])
        prog.AddConstraint(vels[1, 1] == -1*v_contact[1])

        # in the last segment, need to specify bounds for the final position and velocity of the ball
        p_end = p_contact + tau_puck*vels[:,1]*(1-np.exp(-h[1]/tau_puck))   # close form solution for p
        # v_end = vels[:,1]*np.exp(-h[1]/tau_puck)

        p_goal = np.array([self.params.arena_limits_x/2.0, 0])
        prog.AddConstraint(p_end[0] == p_goal[0])
        prog.AddConstraint(p_end[1] == p_goal[1])


        # solve for the periods
        result = Solve(prog)
        if not result.is_success():
            # if debug:
            infeasible = GetInfeasibleConstraints(prog, result)
            print("Infeasible constraints in ContactOptimizer:")
            for i in range(len(infeasible)):
                print(infeasible[i])
            # return directly
            return
        else:
            print("Contact optimization succeeded. Puck velocity: {}".format( result.GetSolution(vels)))
            print("Contact optimization succeeded. h: {}".format( result.GetSolution(h)))
            # print("Contact optimization succeeded. p0: {}".format( result.GetSolution(p0)))
            p_end = p0 + tau_puck*result.GetSolution(vels[:, 0])*(1-np.exp(-result.GetSolution(h[0])/tau_puck))
            print("p_end:{}".format(p_end))


        # return whether successful, and the initial puck velocity
        return result.is_success(), result.GetSolution(vels[:,0])
