import numpy as np
import math
from pydrake.all import eq, DirectCollocation, DirectTranscription, SnoptSolver, Solve, MathematicalProgram, Variable, LinearSystem, GetInfeasibleConstraints

class CentralizedMPC():

    def __init__(self, sim_params, mpc_params):
        self.sim_params = sim_params
        self.mpc_params = mpc_params
        self.prev_u = None
        self.prev_x = None

    def compute_control(self, x_p1, x_p2, x_puck, p_goal, obstacles):
        """Solve for initial velocity for the puck to bounce the wall once and hit the goal."""
        # initialize program
        N = 20 # length of receding horizon
        prog = MathematicalProgram()

        # State and input variables
        x1 = prog.NewContinuousVariables(N+1, 4, name='p1_state')        # state of player 1
        u1 = prog.NewContinuousVariables(N, 2, name='p1_input')          # inputs for player 1
        xp = prog.NewContinuousVariables(N+1, 4, name='puck_state')      # state of the puck

        # Slack variables
        t1_kick = prog.NewContinuousVariables(N+1, name='kick_time')     # slack variables that captures if we player 1 is kicking or not at the given time step
                                                                         # Defined as continous, but treated as mixed integer
        v1_kick = prog.NewContinuousVariables(N+1, 2, name='v1_kick')    # velocity of player after kick to puck
        vp_kick = prog.NewContinuousVariables(N+1, 2, name='vp_kick')    # velocity of puck after being kicked by the player

        #### COST and final states
        # TODO: find adequate final velocity
        x_puck_des = np.concatenate((p_goal, np.zeros(2)), axis=0)      # desired position and vel for puck
        for k in range(N+1):
            prog.AddQuadraticErrorCost(Q=self.mpc_params.Q_puck, x_desired=x_puck_des, vars=xp[k])  # puck in the goal
        for k in range(N):
            prog.AddQuadraticCost(u1[k].flatten().dot(u1[k]))                                       # be wise on control effort

        # Initial states for puck and player
        prog.AddBoundingBoxConstraint(x_p1, x_p1, x1[0])        # Intial state for player 1
        prog.AddBoundingBoxConstraint(x_puck, x_puck, xp[0])    # Initial state for puck

        # Enforce elastic collision for every possible timestep. 
        for k in range(N+1):
            v_puck_bfr = xp[k,2:4]
            v_player_bfr = x1[k, 2:4]
            v_puck_aft, v_player_aft = self.get_reset_velocities(v_puck_bfr, v_player_bfr)
            prog.AddConstraint(eq(vp_kick[k], v_puck_aft))
            prog.AddConstraint(eq(v1_kick[k], v_player_aft))

        # Use slack variable to activate guard condition. 
        M = 15
        for k in range(N+1):
            prog.AddConstraint((x1[k, 0:2]-xp[k, 0:2]).dot(x1[k, 0:2]-xp[k, 0:2]) <= (M*(1-t1_kick[k]))**2)

        # Hybrid dynamics for player
        for k in range(N):
            A = self.mpc_params.A_player
            B = self.mpc_params.B_player
            x1_kick = np.concatenate((x1[k][0:2], v1_kick[k]), axis=0) # The state of the player after it kicks the puck
            x1_next = np.matmul(A, (1 - t1_kick[k])*x1[k] + t1_kick[k]*x1_kick) + np.matmul(B, u1[k])
            prog.AddConstraint(eq(x1[k+1], x1_next))

        # Hybrid dynamics for puck_mass
        for k in range(N):
            A = self.mpc_params.A_puck
            xp_kick = np.concatenate((xp[k][0:2], vp_kick[k]), axis=0) # State of the puck after it gets kicked
            xp_next = np.matmul(A, (1 - t1_kick[k])*xp[k] + t1_kick[k]*xp_kick)
            prog.AddConstraint(eq(xp[k+1], xp_next))

        # Input and arena constraint
        self.add_input_limits(prog, u1, N)
        self.add_arena_limits(prog, x1, N)
        self.add_arena_limits(prog, xp, N)

        # fake mixed-integer constraint
        for k in range(N+1):
            prog.AddBoundingBoxConstraint(0, 1, t1_kick[k])
            prog.AddConstraint(t1_kick[k]*(1-t1_kick[k])==0)
            #prog.AddConstraint(t1_kick[k]==1)

        # solve for the periods
        solver = SnoptSolver()
        result = solver.Solve(prog)

        if not result.is_success():
            print("Unable to find solution.")

        if True:
            print("x1:{}".format(result.GetSolution(x1)))
            print("u1: {}".format( result.GetSolution(u1)))
            print("xp: {}".format( result.GetSolution(xp)))
            print("t1_kick:{}".format(result.GetSolution(t1_kick)))
            print("v1_kick:{}".format(result.GetSolution(v1_kick)))
            print("vp_kick:{}".format(result.GetSolution(vp_kick)))

        # return whether successful, and the initial player velocity
        u1_opt = result.GetSolution(u1)
        return result.is_success(), u1_opt, np.zeros((N, 2))

    def get_reset_velocities(self, v_puck_bfr, v_player_bfr):
        # a: puck
        # b: player
        # 0: before collision
        # f: after colliusion
        ma = self.sim_params.puck_mass
        mb = self.sim_params.player_mass
        va0 = v_puck_bfr
        vb0 = v_player_bfr
        # https://www.khanacademy.org/science/physics/linear-momentum/elastic-and-inelastic-collisions/a/what-are-elastic-and-inelastic-collisions
        vaf = (ma-mb)/(ma+mb)*va0 + 2*mb/(ma+mb)*vb0
        vbf = 2*ma/(ma+mb)*va0 + (mb-ma)/(ma+mb)*vb0
        v_puck_final = vaf
        v_player_final = vbf
        return v_puck_final, v_player_final

    def add_input_limits(self, prog, u, N):
        for k in range(N):
            bound = np.array([self.sim_params.input_limit, self.sim_params.input_limit])
            prog.AddBoundingBoxConstraint(-bound, bound, u[k])

    def add_arena_limits(self, prog, state, N):
        r = self.sim_params.player_radius
        bound = np.array([self.sim_params.arena_limits_x / 2.0 + r, self.sim_params.arena_limits_y / 2.0 + r])
        for k in range(N+1):
            prog.AddBoundingBoxConstraint(-bound, bound, state[k, 0:2])