import numpy as np
import math
from pydrake.all import eq, DirectCollocation, DirectTranscription, SnoptSolver, Solve, MathematicalProgram, Variable, LinearSystem, GetInfeasibleConstraints

class CentralizedMPC():

    def __init__(self, sim_params, mpc_params):
        self.sim_params = sim_params
        self.mpc_params = mpc_params
        self.prev_xp = None

        # To generate intial guess for one player
        self.prev_u = None
        self.prev_x = None

    def compute_control(self, x_p1, x_p2, x_puck, p_goal, obstacles):
        """Solve for initial velocity for the puck to bounce the wall once and hit the goal."""
        # initialize program
        N = self.mpc_params.N # length of receding horizon
        prog = MathematicalProgram()

        # State and input variables
        x1 = prog.NewContinuousVariables(N+1, 4, name='p1_state')        # state of player 1
        u1 = prog.NewContinuousVariables(N, 2, name='p1_input')          # inputs for player 1
        xp = prog.NewContinuousVariables(N+1, 4, name='puck_state')      # state of the puck

        # Slack variables
        t1_kick = prog.NewContinuousVariables(N+1, name='kick_time')     # slack variables that captures if player 1 is kicking or not at the given time step
                                                                         # Defined as continous, but treated as mixed integer. 1 when kicking
        v1_kick = prog.NewContinuousVariables(N+1, 2, name='v1_kick')    # velocity of player after kick to puck
        vp_kick = prog.NewContinuousVariables(N+1, 2, name='vp_kick')    # velocity of puck after being kicked by the player
        dist = prog.NewContinuousVariables(N+1, name='dist_puck_player') # distance between puck and player 
        cost = prog.NewContinuousVariables(N+1, name='cost')             # slack variable to monitor cost

        # Compute distance between puck and player as slack variable. 
        for k in range(N+1):
            r1 = self.sim_params.player_radius
            rp = self.sim_params.puck_radius
            prog.AddConstraint(dist[k] == (x1[k, 0:2]-xp[k, 0:2]).dot(x1[k, 0:2]-xp[k, 0:2]) - (r1+rp)**2)

        #### COST and final states
        # TODO: find adequate final velocity
        x_puck_des = np.concatenate((p_goal, np.zeros(2)), axis=0)      # desired position and vel for puck
        for k in range(N+1):
            Q_dist_puck_goal =10*np.eye(2)
            q_dist_puck_player =0.1
            e1 = x_puck_des[0:2] - xp[k, 0:2] 
            e2 = xp[k, 0:2] - x1[k, 0:2]
            prog.AddConstraint(cost[k] == e1.dot(np.matmul(Q_dist_puck_goal, e1)) + q_dist_puck_player*dist[k])
            prog.AddCost(cost[k])
            #prog.AddQuadraticErrorCost(Q=self.mpc_params.Q_puck, x_desired=x_puck_des, vars=xp[k])  # puck in the goal
            #prog.AddQuadraticErrorCost(Q=np.eye(2), x_desired=x_puck_des[0:2], vars=x1[k, 0:2])
            #prog.AddQuadraticErrorCost(Q=10*np.eye(2), x_desired=np.array([2, 2]), vars=x1[k, 0:2]) # TEST: control position of the player instead of puck
        #for k in range(N):
        #    prog.AddQuadraticCost(1e-2*u1[k].dot(u1[k]))                 # be wise on control effort

        # Initial states for puck and player
        prog.AddBoundingBoxConstraint(x_p1, x_p1, x1[0])        # Intial state for player 1
        prog.AddBoundingBoxConstraint(x_puck, x_puck, xp[0])    # Initial state for puck

        # Compute elastic collision for every possible timestep. 
        for k in range(N+1):
            v_puck_bfr = xp[k,2:4]
            v_player_bfr = x1[k, 2:4]
            v_puck_aft, v_player_aft = self.get_reset_velocities(v_puck_bfr, v_player_bfr)
            prog.AddConstraint(eq(vp_kick[k], v_puck_aft))
            prog.AddConstraint(eq(v1_kick[k], v_player_aft))

        # Use slack variable to activate guard condition based on distance. 
        M = 15**2
        for k in range(N+1): 
            prog.AddLinearConstraint(dist[k] <= M*(1-t1_kick[k]))
            prog.AddLinearConstraint(dist[k] >= - t1_kick[k]*M)

        # Hybrid dynamics for player
        #for k in range(N):
        #    A = self.mpc_params.A_player
        #    B = self.mpc_params.B_player
        #    x1_kick = np.concatenate((x1[k][0:2], v1_kick[k]), axis=0) # The state of the player after it kicks the puck
        #    x1_next = np.matmul(A, (1 - t1_kick[k])*x1[k] + t1_kick[k]*x1_kick) + np.matmul(B, u1[k])
        #    prog.AddConstraint(eq(x1[k+1], x1_next))

        # Assuming player dynamics are not affected by collision
        for k in range(N):
            A = self.mpc_params.A_player
            B = self.mpc_params.B_player
            x1_next = np.matmul(A, x1[k]) + np.matmul(B, u1[k])
            prog.AddConstraint(eq(x1[k+1], x1_next))

        # Hybrid dynamics for puck_mass
        for k in range(N):
            A = self.mpc_params.A_puck
            xp_kick = np.concatenate((xp[k][0:2], vp_kick[k]), axis=0) # State of the puck after it gets kicked
            xp_next = np.matmul(A, (1 - t1_kick[k])*xp[k] + t1_kick[k]*xp_kick)
            prog.AddConstraint(eq(xp[k+1], xp_next))

        # Generate trajectory that is not in direct collision with the puck
        for k in range(N+1):
            eps = 0.1
            prog.AddConstraint(dist[k] >= -eps)

        # Input and arena constraint
        self.add_input_limits(prog, u1, N)
        self.add_arena_limits(prog, x1, N)
        self.add_arena_limits(prog, xp, N)

        # fake mixed-integer constraint
        #for k in range(N+1):
        #    prog.AddConstraint(t1_kick[k]*(1-t1_kick[k])==0)

        # Hot-start
        guess_u1, guess_x1 = self.get_initial_guess(x_p1, p_goal, x_puck[0:2])
        prog.SetInitialGuess(x1, guess_x1)
        prog.SetInitialGuess(u1, guess_u1)
        if not self.prev_xp is None:
            prog.SetInitialGuess(xp, self.prev_xp)
            #prog.SetInitialGuess(t1_kick, np.ones_like(t1_kick))

        # solve for the periods
        # solver = SnoptSolver()
        #result = solver.Solve(prog)

        #if not result.is_success():
        #    print("Unable to find solution.")
        
        # save for hot-start
        #self.prev_xp = result.GetSolution(xp)
        
        #if True:
        #    print("x1:{}".format(result.GetSolution(x1)))
        #    print("u1: {}".format( result.GetSolution(u1)))
        #    print("xp: {}".format( result.GetSolution(xp)))
        #   print('dist{}'.format(result.GetSolution(dist)))
        #    print("t1_kick:{}".format(result.GetSolution(t1_kick)))
        #    print("v1_kick:{}".format(result.GetSolution(v1_kick)))
        #   print("vp_kick:{}".format(result.GetSolution(vp_kick)))
        #    print("cost:{}".format(result.GetSolution(cost)))

        # return whether successful, and the initial player velocity
        #u1_opt = result.GetSolution(u1)
        return True, guess_u1[0,:], np.zeros((2))

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

        ################## approximation for debug
        v_puck_final = v_player_bfr
        v_player_final = v_player_bfr
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

    def get_initial_guess(self, x0, p_goal, p_puck):
        """This is basically the single-agent MPC algorithm"""
        pf, vf = self.get_final_state_for_kick(p_goal, p_puck, 4.0)
        x_des = np.concatenate((pf, vf), axis=0)        
        print("pf", pf)
        print("vf", vf)
        print("p_player", x0[0:2])
        print("p_puck", p_puck)
        print("p_goal", p_goal)
        prog = DirectCollocation(self.mpc_params.sys_c, self.mpc_params.sys_c.CreateDefaultContext(), self.mpc_params.N+1,
                    minimum_timestep=self.mpc_params.minT, maximum_timestep=self.mpc_params.maxT)

        prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
        prog.AddQuadraticErrorCost(Q=self.mpc_params.Omega_N_max, x_desired=x_des, vars=prog.final_state())

        prog.AddEqualTimeIntervalsConstraints()

        # generate trajectory non in collision with puck 
        #for n in range(self.mpc_params.N):
        #    x = prog.state()
        #    eps = 0.1
        #    obs_pos = p_puck[0:2]
        #    prog.AddConstraintToAllKnotPoints((x[0:2]-obs_pos).dot(x[0:2]-obs_pos) >= (self.sim_params.player_radius + self.sim_params.puck_radius - eps)**2)

        prog.AddConstraintToAllKnotPoints(prog.input()[0] <=  self.sim_params.input_limit)
        prog.AddConstraintToAllKnotPoints(prog.input()[0] >= -self.sim_params.input_limit)
        prog.AddConstraintToAllKnotPoints(prog.input()[1] <=  self.sim_params.input_limit)
        prog.AddConstraintToAllKnotPoints(prog.input()[1] >= -self.sim_params.input_limit)

        r = self.sim_params.player_radius
        prog.AddConstraintToAllKnotPoints(prog.state()[0] + r <=  self.sim_params.arena_limits_x / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[0] - r >= -self.sim_params.arena_limits_x / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[1] + r <=  self.sim_params.arena_limits_y / 2.0)
        prog.AddConstraintToAllKnotPoints(prog.state()[1] - r >= -self.sim_params.arena_limits_y / 2.0)

        prog.AddFinalCost(prog.time())

        if not self.prev_u is None and not self.prev_x is None:
            prog.SetInitialTrajectory(traj_init_u=self.prev_u, traj_init_x=self.prev_x)

        solver = SnoptSolver()
        result = solver.Solve(prog)

        u_traj = prog.ReconstructInputTrajectory(result)
        x_traj = prog.ReconstructStateTrajectory(result)

        self.prev_u = u_traj
        self.prev_x = x_traj

        u_vals = u_traj.vector_values(u_traj.get_segment_times())
        x_vals = x_traj.vector_values(x_traj.get_segment_times())

        return u_vals[:, :-1].T, x_vals.T

    def get_final_state_for_kick(self, p_goal, p_puck, kick_velocity):
        shoot_direction = self.get_shoot_direction(p_goal, p_puck)
        pf = p_puck #- shoot_direction*(self.sim_params.puck_radius + self.sim_params.player_radius)
        vf = kick_velocity*shoot_direction
        return pf, vf

    def get_shoot_direction(self, p_goal, p_puck):
        """Returns direction to kick to reach goal"""
        shoot_direction = p_goal - p_puck
        if np.linalg.norm(shoot_direction) > 1e-4:
            shoot_direction/=np.linalg.norm(shoot_direction)
        return shoot_direction
    