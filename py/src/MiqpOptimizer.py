import numpy as np
import matplotlib.pyplot as plt
from src.LinearOptimizer import LinearOptimizer

# pydrake imports
from pydrake.all import MathematicalProgram, OsqpSolver, eq, le, ge
from pydrake.solvers import branch_and_bound

class MiqpOptimizer(LinearOptimizer):
    def __init__(self, params):
        super(MiqpOptimizer, self).__init__(params)
    
    def intercepting_with_obs_avoidance(self, p0, v0, pf, vf, obs, T):
        """Minimum time trajectory that avoids obs"""
        x0 = np.array(np.concatenate((p0, v0), axis=0))
        xf = np.concatenate((pf, vf), axis=0)
        prog = MathematicalProgram()

        # state and control inputs
        N = int(T/self.params.dt) # number of time steps
        state = prog.NewContinuousVariables(N+1, 4, 'state')
        cmd = prog.NewContinuousVariables(N, 2, 'input')

        # Initial and final state
        prog.AddLinearConstraint(eq(state[0], x0))
        prog.AddLinearConstraint(eq(state[-1], xf))
        
        self.add_dynamics(prog, N, state, cmd)

        self.add_input_limits(prog, cmd, N)
        self.add_arena_limits(prog, state, N)

        x_obs = prog.NewBinaryVariables(rows=2, cols=1) # obs x_min, obs x_max
        y_obs = prog.NewBinaryVariables(rows=2, cols=1) # obs y_min, obs y_max

        self.add_puck_as_obstacle(prog, x_obs, y_obs, state, N, obs, vf)

        #prog.AddCost(prog.time())
        for k in range(N):
            prog.AddQuadraticCost(cmd[k].flatten().dot(cmd[k]))

        bb = branch_and_bound.MixedIntegerBranchAndBound(prog, OsqpSolver().solver_id())
        result = bb.Solve()

        if result != result.kSolutionFound:
            raise ValueError('Infeasible optimization problem.')

        u_values = np.array([bb.GetSolution(u) for u in cmd]).T
        print(u_values)
        return result.kSolutionFound, u_values

    def add_dynamics(self, prog, N, state, acc):
        """Add model dynamics to the program as equality constraints for N steps"""
        for k in range(N):
            prog.AddLinearConstraint(eq(state[k+1], np.matmul(self.A, state[k]) + np.matmul(self.B, acc[k])))

    def add_input_limits(self, prog, cmd, N):
        """Add saturation limits to cmd"""
        for k in range(N):
            prog.AddLinearConstraint(le(cmd[k], self.params.input_limit*np.ones(2)))
            prog.AddLinearConstraint(ge(cmd[k], -self.params.input_limit*np.ones(2)))


    def add_arena_limits(self, prog, state, N):
        arena_lims = np.array([self.params.arena_limits_x/2.0, self.params.arena_limits_y/2.0])
        for k in range(N+1):
            prog.AddLinearConstraint(le(state[k][0:2], arena_lims))
            prog.AddLinearConstraint(ge(state[k][0:2], -arena_lims))

                
    def add_puck_as_obstacle(self, prog, x_obs, y_obs,  state, N, p_puck, v_kick):
        """add puck as obstacle using big M notation"""

        # Add constraints with big-M notation
        prog.AddLinearConstraint(x_obs[0][0] + x_obs[1][0] + y_obs[0][0] + y_obs[1][0] >= 1)
        pr_r = self.params.player_radius
        pk_r = self.params.puck_radius
        M = self.params.arena_limits_x
            
        eps = 0.1
        for k in range(N+1):
            if v_kick[0] > 0.0: 
                # make wall on the right of the puck
                prog.AddConstraint(state[k][0] - pr_r >= p_puck[0] + pk_r - M*x_obs[0][0])
                prog.AddConstraint(state[k][0] + pr_r <= p_puck[0] + pk_r - eps + M*x_obs[1][0])
            else:
                # make wall on the left of the puck
                prog.AddConstraint(state[k][0] - pr_r >= p_puck[0] - pk_r + eps - M*x_obs[0][0])
                prog.AddConstraint(state[k][0] + pr_r <= p_puck[0] - pk_r + M*x_obs[1][0])

            if v_kick[1] > 0.0:
                # make wall above the puck
                prog.AddConstraint(state[k][1] - pr_r >= p_puck[1] + pk_r - M*y_obs[0][0])
                prog.AddConstraint(state[k][1] + pr_r <= p_puck[1] + pk_r - eps + M*y_obs[1][0])
            else:
                # make wall below the puck
                prog.AddConstraint(state[k][1] - pr_r >= p_puck[1] - pk_r + eps - M*y_obs[0][0])
                prog.AddConstraint(state[k][1] + pr_r <= p_puck[1] - pk_r + M*y_obs[1][0])





