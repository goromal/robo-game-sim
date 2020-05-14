import numpy as np
import math
import copy
from pydrake.all import eq, le, ge,  DirectCollocation, DirectTranscription, SnoptSolver, Solve, MathematicalProgram, Variable, LinearSystem, GetInfeasibleConstraints, GurobiSolver

import matplotlib.pyplot as plt


class MpcParams():
    def __init__(self, params):
        self.params = params

        self.A_player = np.eye(4) + self.params.dt*np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_player, 0], [0, 0, 0, -1.0/self.params.tau_player]])
        self.B_player = self.params.dt*np.array([[0, 0], [0, 0], [1.0/self.params.tau_player, 0], [0, 1.0/self.params.tau_player]])
        self.C_player = np.eye(4)
        self.D_player = np.zeros((4,2))
        self.sys_player = LinearSystem(self.A_player, self.B_player, self.C_player, self.D_player, self.params.dt)

        self.A_puck = np.eye(4) + self.params.dt*np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_puck, 0], [0, 0, 0, -1.0/self.params.tau_puck]])
        self.B_puck = np.zeros((4, 2))
        self.C_puck = np.eye(4)
        self.D_puck = np.zeros((4,2))
        self.sys_puck = LinearSystem(self.A_puck, self.B_puck, self.C_puck, self.D_puck, self.params.dt)

        self.Q_puck = np.eye(4) # TODO: penalize velocity differently
        self.N = 50
        self.minT = self.params.dt/(self.N + 1)
        self.maxT = 4.0*self.params.dt
        self.Omega_N_max = np.array([[10.0,  0.0,  0.0,  0.0],[ 0.0, 10.0,  0.0,  0.0],[ 0.0,  0.0, 20.0,  0.0],[ 0.0,  0.0,  0.0, 20.0]])
        #self.Q_puck[2:4, 2:4] = np.zeros((2,2))

class JointPuckPlayerMPC():

    def __init__(self, sim_params):
        self.sim_params = sim_params
        self.mpc_params = MpcParams(sim_params)

        # To generate intial guess for one player
        self.prev_u1 = None
        self.prev_x1 = None
        self.prev_xp = None

        self.fig_idx = 0

    def compute_control(self, x_p1, x_puck, p_goal, obstacles = None):
        """Generata player's trajectory that drives the puck into the goal"""
        
        # initialize program
        N = self.mpc_params.N # length of horizon
        prog = MathematicalProgram()

        # State and input variables
        x1 = prog.NewContinuousVariables(N+1, 4, name='p1_state')        # state of player 1
        u1 = prog.NewContinuousVariables(N, 2, name='p1_input')          # inputs for player 1
        xp = prog.NewContinuousVariables(N+1, 4, name='puck_state')      # state of the puck
        v1_kick = prog.NewContinuousVariables(N+1, 2, name='v1_kick')    # velocity of player after kick to puck
        vp_kick = prog.NewContinuousVariables(N+1, 2, name='vp_kick')    # velocity of puck after being kicked by the player

        # Slack variables
        t1_kick = prog.NewContinuousVariables(N+1, name='kick_time')     # slack variables that captures if player 1 is kicking or not at the given time step
                                                                         # Defined as continous, but treated as mixed integer. 1 when kicking
        lambda_1 = prog.NewContinuousVariables(N+1, 2, name='lambda_1')  # Contact force between player and puck
        dist_p_1 = prog.NewContinuousVariables(N+1, name='dist_p_1_puck_player') # distance between puck and player 
        cost = prog.NewContinuousVariables(N+1, name='cost')             # slack variable to monitor cost

        # Compute distance between puck and player as slack variable. 
        for k in range(N+1):
            r1 = self.sim_params.player_radius
            rp = self.sim_params.puck_radius
            prog.AddConstraint(dist_p_1[k] == (x1[k, 0:2]-xp[k, 0:2]).dot(x1[k, 0:2]-xp[k, 0:2]) - (r1+rp)**2)

        # Enforce distance to be positive
        for k in range(N+1):
            prog.AddConstraint(dist_p_1[k] >= 0)

        # Enfore t kick to be 0 at the beginning
        prog.AddConstraint(t1_kick[0] == 0)

        # COST: minimize distance of the puck from the goal
        x_puck_des = np.concatenate((p_goal, np.zeros(2)), axis=0)      # desired position and vel for puck
        for k in range(N+1):
            Q_dist_puck_goal = 10.0*np.eye(2)
            q_dist_puck_player = 0.0
            e1 = x_puck_des[0:2] - xp[k, 0:2] 
            e2 = xp[k, 0:2] - x1[k, 0:2]
            prog.AddConstraint(cost[k] == e1.dot(np.matmul(Q_dist_puck_goal, e1))+ q_dist_puck_player*dist_p_1[k])
            prog.AddCost(cost[k])

        # Add small penalty on control input
        R = 0.1*np.eye(2)
        for k in range(N):
            prog.AddCost(u1[k].dot(np.matmul(R, u1[k])))

        # Initial states for puck and player_
        prog.AddBoundingBoxConstraint(x_p1, x_p1, x1[0])        # Intial state for player 1
        prog.AddBoundingBoxConstraint(x_puck, x_puck, xp[0])    # Initial state for puck

        # Use slack variable to activate guard condition based on distance. 
        M = 15**2
        for k in range(N+1): 
            prog.AddLinearConstraint(dist_p_1[k] <= M*(1-t1_kick[k]))

        # Apply constraints on the impact force based on the guard. 
        M = np.ones((2,1))*10
        for k in range(N+1):
            prog.AddConstraint(le(lambda_1[k], M*(t1_kick[k]))) # lambda <= M*(1-t_kick)
            prog.AddConstraint(ge(lambda_1[k], -M*(t1_kick[k])))
        
        # Compute elastic collision for every possible timestep. 
        for k in range(N+1):
            v_puck_bfr = xp[k,2:4]
            v_player_bfr = x1[k, 2:4]
            v_puck_aft, v_player_aft = self.get_reset_velocities(v_puck_bfr, v_player_bfr)
            prog.AddConstraint(eq(vp_kick[k], v_puck_aft))
            prog.AddConstraint(eq(v1_kick[k], v_player_aft))

        # Enforce reset condition at impact using the BIGM method for the puck and the player
        for k in range(N):
            
            M = 20*np.ones(2)
            # puck
            prog.AddLinearConstraint(le(vp_kick[k+1] - xp[k+1, 2:4], M*(1-t1_kick[k]))) # puck
            prog.AddLinearConstraint(ge(vp_kick[k+1] - xp[k+1, 2:4], -M*(1-t1_kick[k]))) # puck 

            # player
            prog.AddLinearConstraint(le(v1_kick[k+1] - x1[k+1, 2:4], M*(1-t1_kick[k]))) # player
            prog.AddLinearConstraint(ge(v1_kick[k+1] - x1[k+1, 2:4], -M*(1-t1_kick[k])))

        # Player dynamics with collisions
        for k in range(N):
            A_1 = self.mpc_params.A_player
            B_1 = self.mpc_params.B_player
            #E_1 = np.ones((2,1))*self.sim_params.dt/self.sim_params.player_mass
            x1_next = np.matmul(A_1, x1[k]) + np.matmul(B_1, u1[k]) + np.matmul(B_1, lambda_1[k])
            prog.AddConstraint(eq(x1[k+1], x1_next))

        # Puck dynamics with collisions
        for k in range(N):
            A_p = self.mpc_params.A_puck
            E_p = self.sim_params.dt*np.array([[0, 0], [0, 0], [1.0/self.sim_params.tau_puck, 0], [0, 1.0/self.sim_params.tau_puck]])
            #xp_kick = np.concatenate((xp[k][0:2], vp_kick[k]), axis=0) # State of the puck after it gets kicked
            xp_next = np.matmul(A_p, xp[k]) + np.matmul(E_p, -lambda_1[k])
            prog.AddConstraint(eq(xp[k+1], xp_next))
        
        # Input and arena constraint
        self.add_input_limits(prog, u1, N)
        self.add_arena_limits(prog, x1, N)
        self.add_arena_limits(prog, xp, N)

        prog.AddLinearConstraint(sum(t1_kick) >= 1)

        # fake mixed-integer constraint
        for k in range(N+1):
            prog.AddConstraint(t1_kick[k]*(1-t1_kick[k])==0)

        # Hot-start from DMPC
        #if False:
        #    guess_u1, guess_x1 = self.get_initial_guess(x_p1, p_goal, x_puck[0:2])
        #    prog.SetInitialGuess(x1, guess_x1)
        #    prog.SetInitialGuess(u1, guess_u1)
        
        if not (self.prev_xp is None):
            prog.SetInitialGuess(xp, self.prev_xp)
            prog.SetInitialGuess(x1, self.prev_x1)
            prog.SetInitialGuess(u1, self.prev_u1)

        # solve for the periods
        solver = SnoptSolver()
        result = solver.Solve(prog)

        if not result.is_success():
            print("Unable to find solution.")
        
        # save for hot-start
        self.prev_xp = result.GetSolution(xp)
        self.prev_u1  = result.GetSolution(u1)
        self.prev_x1 = result.GetSolution(x1)
        
        if True:
            print("State of player 1 x1:{}".format(result.GetSolution(x1)))
            print("Commands of player 1 u1: {}".format(result.GetSolution(u1)))
            print("State of puck xp: {}".format(result.GetSolution(xp)))
            print("dist_p_1ance of puck from the goal: {}".format(result.GetSolution(dist_p_1)))
            print("Is player kicking? t1_kick:{}".format(result.GetSolution(t1_kick)))
            print("COST (distance of puck from goal):{}".format(result.GetSolution(cost)))
            print("Contact force: {}".format(result.GetSolution(lambda_1)))
            plt.ioff()
            plt.figure()
            fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(7, 1, figsize=(8,20))
            ax0.plot(result.GetSolution(cost))
            ax0.set_title("Cost")
            ax1.plot(result.GetSolution(t1_kick))
            ax1.set_title("Kick")
            ax2.plot(result.GetSolution(lambda_1))
            ax2.set_title("Contact force")
            ax3.plot(result.GetSolution(dist_p_1))
            ax3.set_title("Distance puck player")
            ax4.plot(result.GetSolution(u1))
            ax4.set_title("Actuation (u)")
            ax5.plot(result.GetSolution(x1))
            ax5.set_title("Player states")
            ax6.plot(result.GetSolution(xp))
            ax6.set_title("Puck states")
            plt.savefig("fig_" + str(self.fig_idx) + ".pdf")
            self.fig_idx += 1
            
        # return whether successful, and the initial player velocity
        u1_opt = result.GetSolution(u1)
        return u1_opt, np.zeros(2)

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
        #v_puck_final = v_player_bfr
        #v_player_final = v_player_bfr
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

    #def compute_control(self, x_p1, x_p2, x_puck, p_goal, obstacles):
    #    u1 = self.get_initial_guess(x_p1, p_goal, x_puck[0:2], obstacles)
    #    print(u1.shape)
    #    return True, u1, np.zeros(2)

    def get_initial_guess(self, x_p1, p_goal, p_puck, obstacles):
        """This is basically the single-agent MPC algorithm"""
        hit_dir = p_goal - p_puck
        hit_dir = 6.0 * hit_dir / np.linalg.norm(hit_dir)
        x_des = np.array([p_puck[0], p_puck[1], hit_dir[0], hit_dir[1]])
        #x_des = np.array([1.0, 1.0, 0, 0])    
        print("x_des: {}, {}".format(x_des[0], x_des[1]))
        print("x_des shape", x_des.shape)
        print("zeros.shape", np.zeros(4).shape)
        print("p_player", x_p1[0:2])
        print("p_puck {}, {}".format(p_puck[0], p_puck[1]))
        print("p_goal", p_goal)
        prog = DirectCollocation(self.mpc_params.sys_c, self.mpc_params.sys_c.CreateDefaultContext(), self.mpc_params.N+1,
                    minimum_timestep=self.mpc_params.minT, maximum_timestep=self.mpc_params.maxT)

        prog.AddBoundingBoxConstraint(x_p1, x_p1, prog.initial_state())
        prog.AddQuadraticErrorCost(Q=self.mpc_params.Omega_N_max, x_desired=x_des, vars=prog.final_state())

        prog.AddEqualTimeIntervalsConstraints()

        # generate trajectory non in collision with puck 
        #for n in range(self.mpc_params.N):
        #    x = prog.state()
        #    eps = 0.1
        #    obs_pos = p_puck[0:2]
        #    prog.AddConstraintToAllKnotPoints((x[0:2]-obs_pos).dot(x[0:2]-obs_pos) >= (self.sim_params.player_radius + self.sim_params.puck_radius - eps)**2)

        for obs_pos in obstacles:
            for n in range(self.mpc_params.N):
                x = prog.state()
                prog.AddConstraintToAllKnotPoints((x[0:2]-obs_pos).dot(x[0:2]-obs_pos) >= (2.0*self.sim_params.player_radius)**2)

        
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
        print(u_vals)
        print(u_vals[:,0])
        return u_vals[:,0]

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
    