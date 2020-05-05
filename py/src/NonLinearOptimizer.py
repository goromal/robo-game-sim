import numpy as np
import matplotlib.pyplot as plt
from src.LinearOptimizer import LinearOptimizer

# pydrake imports
from pydrake.all import MathematicalProgram, OsqpSolver, SnoptSolver, eq, le, ge, DirectCollocation
from pydrake.solvers import branch_and_bound

class NonLinearOptimizer(LinearOptimizer):
    def __init__(self, params):
        super(NonLinearOptimizer, self).__init__(params)
    
    def intercepting_with_obs_avoidance(self, p0, v0, pf, vf, T, obstacles=None, p_puck=None):
        """Intercepting trajectory that avoids obs"""
        x0 = np.array(np.concatenate((p0, v0), axis=0))
        xf = np.concatenate((pf, vf), axis=0)
        prog = MathematicalProgram()

        # state and control inputs
        N = int(T/self.params.dt) # number of time steps
        state = prog.NewContinuousVariables(N+1, 4, 'state')
        cmd = prog.NewContinuousVariables(N, 2, 'input')

        # Initial and final state
        prog.AddLinearConstraint(eq(state[0], x0))
        #prog.AddLinearConstraint(eq(state[-1], xf))
        prog.AddQuadraticErrorCost(Q=10.0*np.eye(4), x_desired=xf, vars=state[-1])
        
        self.add_dynamics(prog, N, state, cmd)

        ## Input saturation
        self.add_input_limits(prog, cmd, N)

        # Arena constraints
        self.add_arena_limits(prog, state, N)

        for k in range(N):
            prog.AddQuadraticCost(cmd[k].flatten().dot(cmd[k]))

        # Add non-linear constraints - will solve with SNOPT
        # Avoid other players
        if obstacles != None:
            for obs in obstacles:
                self.avoid_other_player_nl(prog, state, obs, N)

        # avoid hitting the puck while generating a kicking trajectory
        if not p_puck.any(None):
            self.avoid_puck_nl(prog, state, N, p_puck)

        solver = SnoptSolver()
        result = solver.Solve(prog)
        solution_found = result.is_success()
        if not solution_found:
            print("Solution not found for intercepting_with_obs_avoidance")

        u_values = result.GetSolution(cmd)
        return solution_found, u_values.T

    def intercepting_with_obs_avoidance_bb(self, p0, v0, pf, vf, T, obstacles=None, p_puck=None):
        """kick while avoiding obstacles using big M formulation and branch and bound"""
        x0 = np.array(np.concatenate((p0, v0), axis=0))
        xf = np.concatenate((pf, vf), axis=0)
        prog = MathematicalProgram()

        # state and control inputs
        N = int(T/self.params.dt) # number of command steps
        state = prog.NewContinuousVariables(N+1, 4, 'state')
        cmd = prog.NewContinuousVariables(N, 2, 'input')

        # Initial and final state
        prog.AddLinearConstraint(eq(state[0], x0))
        prog.AddLinearConstraint(eq(state[-1], xf))
        
        self.add_dynamics(prog, N, state, cmd)

        ## Input saturation
        self.add_input_limits(prog, cmd, N)

        # Arena constraints
        self.add_arena_limits(prog, state, N)

        # Add Mixed-integer constraints - will solve with BB

        # avoid hitting the puck while generating a kicking trajectory
        # MILP formulation with B&B solver
        x_obs_puck = prog.NewBinaryVariables(rows=N+1, cols=2) # obs x_min, obs x_max
        y_obs_puck = prog.NewBinaryVariables(rows=N+1, cols=2) # obs y_min, obs y_max
        self.avoid_puck_bigm(prog, x_obs_puck, y_obs_puck, state, N, p_puck)

        # Avoid other players
        if obstacles != None:
            x_obs_player = list()
            y_obs_player = list()
            for i, obs in enumerate(obstacles):
                x_obs_player.append(prog.NewBinaryVariables(rows=N+1, cols=2)) # obs x_min, obs x_max
                y_obs_player.append(prog.NewBinaryVariables(rows=N+1, cols=2)) # obs y_min, obs y_max
                self.avoid_other_player_bigm(prog, x_obs_player[i], y_obs_player[i], state, obs, N)

        # Solve with simple b&b solver
        for k in range(N):
            prog.AddQuadraticCost(cmd[k].flatten().dot(cmd[k]))

        bb = branch_and_bound.MixedIntegerBranchAndBound(prog, OsqpSolver().solver_id())
        result = bb.Solve()
        if result != result.kSolutionFound:
            raise ValueError('Infeasible optimization problem.')
        u_values = np.array([bb.GetSolution(u) for u in cmd]).T
        solution_found = result.kSolutionFound
        return solution_found, u_values

    def min_time_traj_avoid_obs(self, p0, v0, pf, vf, obstacles=None, p_puck=None):
        """Minimum time trajectory while avoiding obstacles."""
        x0 = np.array(np.concatenate((p0, v0), axis=0))
        xf = np.concatenate((pf, vf), axis=0)

        N = 20
        prog = DirectCollocation(self.sys_c, 
                                 self.sys_c.CreateDefaultContext(), 
                                 N, minimum_timestep=self.params.dt, 
                                 maximum_timestep=self.params.dt)

        # Initial and final state
        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        prog.AddQuadraticErrorCost(Q=np.eye(4), x_desired=xf, vars=prog.final_state())
        u =  prog.input()
        prog.AddRunningCost(0.1*u.dot(u))

        prog.AddEqualTimeIntervalsConstraints()
    
        ## Input saturation
        self.add_input_limits(prog)

        # Arena constraints
        self.add_arena_limits(prog)

        prog.AddFinalCost(prog.time())

        # Add non-linear constraints - will solve with SNOPT
        # Avoid other players
        if obstacles != None:
            for p_obs in obstacles:
                distance = prog.state()[0:2] - p_obs
                prog.AddConstraintToAllKnotPoints(distance.dot(distance) >= (2.0*self.params.player_radius)**2)

        # avoid hitting the puck while generating a kicking trajectory
        #if not p_puck.any(None):
        #    distance = prog.state()[0:2] - p_puck
        #    prog.AddConstraintToAllKnotPoints(distance.dot(distance) >= (self.params.player_radius + self.params.puck_radius)**2)

        solver = SnoptSolver()
        result = solver.Solve(prog)
        solution_found = result.is_success()
        if not solution_found:
            print("Solution not found for intercepting_with_obs_avoidance")

        u_traj = prog.ReconstructInputTrajectory(result)
        u_values = u_traj.vector_values(u_traj.get_segment_times())
        return solution_found, u_values

    def add_dynamics(self, prog, N, state, acc):
        """Add model dynamics to the program as equality constraints for N steps"""
        for k in range(N):
            prog.AddLinearConstraint(eq(state[k+1], np.matmul(self.A, state[k]) + np.matmul(self.B, acc[k])))

    def add_input_limits_nl(self, prog, cmd, N):
        """Add saturation limits to cmd"""
        for k in range(N):
            prog.AddLinearConstraint(le(cmd[k], self.params.input_limit*np.ones(2)))
            prog.AddLinearConstraint(ge(cmd[k], -self.params.input_limit*np.ones(2)))

    def add_arena_limits_nl(self, prog, state, N):
        arena_lims = np.array([self.params.arena_limits_x/2.0, self.params.arena_limits_y/2.0])
        for k in range(N+1):
            prog.AddLinearConstraint(le(state[k][0:2], arena_lims))
            prog.AddLinearConstraint(ge(state[k][0:2], -arena_lims))

    ############################################################
    #####    Obstacle avoidance for non-linear formulation  
    ############################################################

    def avoid_other_player_nl(self, prog, state, p_other_player, N):
        """avoid other player, assuming the player remains where it is, using non-linear constraint"""
        eps = 0.0
        for k in range(N+1):
            distance = state[k][0:2] - p_other_player
            prog.AddConstraintToAllKnotPoints(distance.dot(distance) >= (2.0*self.params.player_radius + eps)**2)

    def avoid_puck_nl(self, prog, state, N, p_puck):
        """avoid the puck (but allow kick)"""
        eps = 0.2 # tunable parameter for tighter/more relaxed tolerance
        for k in range(N+1):
            distance = state[k][0:2] - p_puck
            prog.AddConstraintToAllKnotPoints(distance.dot(distance) >= (self.params.player_radius + self.params.puck_radius - eps)**2)
    
    ############################################################
    #####    Obstacle avoidance for MIQP formulation  
    ############################################################

    def avoid_other_player_bigm(self, prog, x_obs, y_obs, state, p_other_player, N):
        """Avoid other player using bigM formulation """
        obstacle_size = 2.0*self.params.player_radius
        self.avoid_obstacle_bigm(prog, x_obs, y_obs,  state, N, p_other_player, obstacle_size)
                
    def avoid_puck_bigm(self, prog, x_obs, y_obs,  state, N, p_puck):
        """add puck as obstacle using big M notation"""
        epsilon = 0.2 # Reduce obstacle size to allow kicking of the puck
        obstacle_size = self.params.player_radius + self.params.puck_radius - epsilon
        self.avoid_obstacle_bigm(prog, x_obs, y_obs,  state, N, p_puck, obstacle_size)

    def avoid_obstacle_bigm(self, prog, x_obs, y_obs,  state, N, p_obs, obs_size):
        """Add constraint to avoid generic obstacle using the big-M notation"""
        Mx = 2.0*self.params.arena_limits_x
        My = 2.0*self.params.arena_limits_y
        for k in range(N+1):
            prog.AddConstraint(x_obs[k][0] + x_obs[k][1] + y_obs[k][0] + y_obs[k][1] == 1)
            prog.AddConstraint(state[k][0] >= p_obs[0] + obs_size - Mx*(1 - x_obs[k][0]))
            prog.AddConstraint(state[k][0] <= p_obs[0] - obs_size + Mx*(1 - x_obs[k][1]))
            prog.AddConstraint(state[k][1] >= p_obs[1] + obs_size - My*(1 - y_obs[k][0]))
            prog.AddConstraint(state[k][1] <= p_obs[1] - obs_size + My*(1 - y_obs[k][1]))



