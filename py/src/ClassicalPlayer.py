import numpy as np
from src.LinearOptimizer import LinearOptimizer
from src.ContactOptimizer import ContactOptimizer
from src.NonLinearOptimizer import NonLinearOptimizer

class ClassicalPlayer:
    def __init__(self, params, field, team, player_id):
        # player parameters
        self.params = params
        self.field = field          # on which side of the arena the player plays (defend) -1 left, +1 right
        self.team = team            # A or B
        self.player_id = player_id  # 1 or 2

        # Latest generated trajectory and time index associated with it
        self.u_traj = np.zeros((2,1))
        self.t_idx = 0 # index at which the trajectory needs to be evaluated

        # Optimizers
        self.linear_optimizer = LinearOptimizer(self.params)
        self.contact_optimizer = ContactOptimizer(self.params)
        self.miqp_optimizer = NonLinearOptimizer(self.params)

    # Return latest control action and
    # percentage of completion of current action.
    # Has to be called at 1/dt rate.
    def get_control(self):

        percentage_completed = self.task_percentage_completed() # 1.0 when current trajectory is completed

        if percentage_completed <= 1.0:
            u_cmd = self.u_traj[:, self.t_idx]
            self.t_idx += 1
            return u_cmd, percentage_completed

        # if we are here is because we have completed current trajectory....
        # Let's go idle
        self.idle()
        return self.u_traj[:,self.t_idx], 1.0


    ###########################################################################
    ### All the methods below here generate a trajectory (of length one or more)
    ### Updating the self.u_traj and setting self.t_idx = 0
    ###########################################################################

    # Generate trajectory and stores it in the class state
    # Returns if optimization was successful or not
    def timed_kick(self, state, kick_velocity, time_to_kick):
        """Kick the puck towards the goal with the specified speed within the given time."""
        p_puck = state.get_puck_pos()
        p_goal = self.get_adversary_goal_pos()

        p0 = state.get_player_pos(self.team, self.player_id)
        v0 = state.get_player_vel(self.team, self.player_id)
        pf, vf = self.get_final_state_for_kick(p_goal, p_puck, kick_velocity)
        T = time_to_kick

        # Store trajectory and reset execution timer
        successful, self.u_traj = self.linear_optimizer.intercepting_traj(p0, v0, pf, vf, T)
        self.t_idx = 0

        return successful


    def timed_kick_avoiding_obs(self, state, kick_velocity, time_to_kick):
        """Finite-time kick while avoiding other players and puck"""
        p_puck = state.get_puck_pos()
        p_goal = self.get_adversary_goal_pos()

        p0 = state.get_player_pos(self.team, self.player_id)
        v0 = state.get_player_vel(self.team, self.player_id)
        pf, vf = self.get_final_state_for_kick(p_goal, p_puck, kick_velocity)

        # define obstacles to avoid:
        other_players = self.get_pos_of_other_players(state)

        successful, self.u_traj = self.miqp_optimizer.intercepting_with_obs_avoidance(p0, v0, pf, vf, time_to_kick, other_players, p_puck)
        #successful, self.u_traj = self.miqp_optimizer.intercepting_with_obs_avoidance_bb(p0, v0, pf, vf, time_to_kick, other_players, p_puck)
        self.t_idx = 0

        return successful

    def idle(self):
        """player stays where it is"""
        self.u_traj = np.zeros((2, 1))
        self.t_idx = 0

        # return successfully generated
        return True

    def simple_kick(self, state, kick_velocity):
        """Minimum time trajectory with desired final velocity pointing towards goal"""
        p_puck = state.get_puck_pos()
        p_goal = self.get_adversary_goal_pos()

        p0 = state.get_player_pos(self.team, self.player_id)
        v0 = state.get_player_vel(self.team, self.player_id)
        pf, vf = self.get_final_state_for_kick(p_goal, p_puck, kick_velocity)

        # Store trajectory and reset execution timer
        successful, u_traj = self.linear_optimizer.min_time_traj(p0, v0, pf, vf)
        if successful and (len(u_traj) is not 0):
            self.u_traj = u_traj
            self.t_idx = 0

        return successful

    def simple_kick_avoiding_obs(self, state, kick_velocity):
        """Minimum time trajectory with desired final velocity pointing towards goal while avoiding obstacle"""
        p_puck = state.get_puck_pos()
        p_goal = self.get_adversary_goal_pos()

        p0 = state.get_player_pos(self.team, self.player_id)
        v0 = state.get_player_vel(self.team, self.player_id)
        pf, vf = self.get_final_state_for_kick(p_goal, p_puck, kick_velocity)

        # define obstacles to avoid:
        other_players = self.get_pos_of_other_players(state)

        # Store trajectory and reset execution timer
        successful, u_traj = self.miqp_optimizer.min_time_traj_avoid_obs(p0, v0, pf, vf, other_players, p_puck)

        if successful and (len(u_traj) is not 0):
            self.u_traj = u_traj
            self.t_idx = 0

        return successful

    def bounce_kick(self, state, which_wall):
        """Kick the puck to the specified wall and bounce to adversary's goal. Require that tau_puck >= 1."""
        puck_pos = state.get_puck_pos()
        p_goal = self.get_adversary_goal_pos()
        successful, v_puck_desired = self.contact_optimizer.bounce_pass_wall(puck_pos, p_goal, which_wall)
        if successful:
            p0 = state.get_player_pos(self.team, self.player_id)
            v0 = state.get_player_vel(self.team, self.player_id)
            p0_puck = state.get_puck_pos()
            v0_puck = state.get_puck_vel()
            successful, u_traj = self.linear_optimizer.min_time_bounce_kick_traj(p0, v0, p0_puck, v0_puck, v_puck_desired)
            # successful, self.u_traj = self.linear_optimizer.min_time_bounce_kick_traj_dir_col(p0, v0, p0_puck, v0_puck, v_puck_desired) # Not working

            if successful:
                self.u_traj = u_traj
                self.t_idx = 0
                # print("min_time_bounce_kick_traj optimization succeeded.")

        return successful

    def defend_kick(self, state, kick_vel):
        """Kick the ball in opponent's open field"""
        p_puck = state.get_puck_pos()

        p0 = state.get_player_pos(self.team, self.player_id)
        v0 = state.get_player_vel(self.team, self.player_id)

        # get opponents' positions
        opp_pos1 = state.get_player_pos(self.get_adversary_team(), 1)
        opp_pos2 = state.get_player_pos(self.get_adversary_team(), 2)
        shoot_direction = self.get_normalized_vector([ -self.field, np.sign(opp_pos1[1] + opp_pos2[1])])

        vf = shoot_direction * kick_vel
        pf = p_puck - shoot_direction*(self.params.puck_radius + self.params.player_radius)

        # Store trajectory and reset execution timer
        successful, u_traj = self.linear_optimizer.min_time_traj(p0, v0, pf, vf)
        if successful and (len(u_traj) is not 0):
            self.u_traj = u_traj
            self.t_idx = 0

        return successful

    def defend(self, state):
        """Stay in between puck and home goal, within the goalie region."""
        p0 = state.get_player_pos(self.team, self.player_id)
        v0 = state.get_player_vel(self.team, self.player_id)
        pf = self.get_home_goal_pos() + 0.5* (state.get_puck_pos() - self.get_home_goal_pos())

        successful, u_traj = self.linear_optimizer.min_time_traj(p0, v0, pf, np.zeros(2))
        if successful:
            self.u_traj = u_traj
            self.t_idx = 0

        return successful

    ###########################################################################
    ### Helper functions
    ###########################################################################


    # Check if player is free or busy executing some long open-loop actions
    def is_idle(self):
        if self.task_percentage_completed() >= 1.0:
            return True
        else :
            return False

    # percentage of task completion
    def task_percentage_completed(self):
        return (self.t_idx+1)/len(self.u_traj[0,:]) # > 1.0 when current trajectory is completed

    def get_adversary_goal_pos(self):
        """Where the goal should be kicked."""
        if self.field > 0:
            return np.array([-self.params.arena_limits_x/2.0, 0.0])
        else :
            return np.array([self.params.arena_limits_x/2.0, 0.0])

    def get_home_goal_pos(self):
        """Where the goal should be kicked."""
        if self.field < 0:
            return np.array([-self.params.arena_limits_x/2.0, 0.0])
        else :
            return np.array([self.params.arena_limits_x/2.0, 0.0])

    def get_final_state_for_kick(self, p_goal, p_puck, kick_velocity):
        """Get desired final position and velocity for the player to shoot the goal."""
        shoot_direction = self.get_shoot_direction(p_goal, p_puck)
        pf = p_puck - shoot_direction*(self.params.puck_radius + self.params.player_radius)
        vf = kick_velocity*shoot_direction
        return pf, vf

    def get_shoot_direction(self, p_goal, p_puck):
        """Returns direction to kick to reach goal"""
        shoot_direction = p_goal - p_puck
        if np.linalg.norm(shoot_direction) > 1e-4:
            shoot_direction/=np.linalg.norm(shoot_direction)
        return shoot_direction

    def get_pos_of_other_players(self, state):
        """Get positions of all other players."""
        positions = list()
        positions.append(state.get_player_pos(self.team, self.get_teammate_id()))
        positions.append(state.get_player_pos(self.get_adversary_team(), 1))
        positions.append(state.get_player_pos(self.get_adversary_team(), 2))
        return positions

    def get_adversary_team(self):
        """Returns the team adversary to the player's team"""
        if self.team == "A":
            return "B"
        elif self.team == "B":
            return "A"
        else:
            raise Exception("Team not recognized! Team can either be \"A\" or \"B\"")

    def get_teammate_id(self):
        """Returns the other team mate of the current player"""
        if self.player_id == 1:
            return 2
        elif self.player_id == 2:
            return 1
        else:
            raise Exception("self.player_id not recognizer! player_id can ether be 1 or 2")

    def get_normalized_vector(self, v):
        """Get normalized vector."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
