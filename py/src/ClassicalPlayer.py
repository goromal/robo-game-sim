import numpy as np
from src.LinearOptimizer import LinearOptimizer
from src.ContactOptimizer import ContactOptimizer

class ClassicalPlayer:
    def __init__(self, params, field, team, player_id, state):
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

    # Check if player is free or busy executing some long open-loop actions
    def is_idle(self):
        if self.task_percentage_completed() >= 1.0:
            return True
        else :
            return False
        
    # percentage of task completion 
    def task_percentage_completed(self):
        return (self.t_idx+1)/len(self.u_traj[0,:]) # > 1.0 when current trajectory is completed

    ###########################################################################
    ### All the methods below here generate a trajectory (of length one or more)
    ### Updating the self.u_traj and setting self.t_idx = 0

    # Generate trajectory and stores it in the class state
    # Returns if optimization was successfull or not
    def timed_kick(self, state, kick_velocity, time_to_kick):
        p_puck = state.get_puck_pos()
        p_goal = self.get_adversary_goal_pos()

        shoot_direction = self.get_shoot_direction(p_goal, p_puck)

        # Jeremy's timed kick
        p0 = state.get_player_pos(self.team, self.player_id)
        v0 = state.get_player_vel(self.team, self.player_id)
        pf = p_puck - shoot_direction*(self.params.puck_radius + self.params.player_radius)
        vf = kick_velocity*shoot_direction
        T = time_to_kick
  
        # Store trajectory and reset execution timer
        successfull, self.u_traj = self.linear_optimizer.intercepting_traj(p0, v0, pf, vf, T)
        self.t_idx = 0

        return successfull
    
    # player stays where it is
    def idle(self):

        self.u_traj = np.zeros((2, 1))
        self.t_idx = 0

        # return successfully generated
        return True

    # TODO tries to reach and hit the ball in minimum time
    # with maximum possible final velocity
    def simple_kick(self, state, kick_velocity):
        """Minimum time trajectory with desired final velocity pointing towards goal"""
        p_puck = state.get_puck_pos()
        p_goal = self.get_adversary_goal_pos()

        shoot_direction = self.get_shoot_direction(p_goal, p_puck)

        # Jeremy's timed kick
        p0 = state.get_player_pos(self.team, self.player_id)
        v0 = state.get_player_vel(self.team, self.player_id)
        pf = p_puck - shoot_direction*(self.params.puck_radius + self.params.player_radius)
        vf = kick_velocity*shoot_direction
  
        # Store trajectory and reset execution timer
        successfull, self.u_traj = self.linear_optimizer.min_time_traj(p0, v0, pf, vf)
        self.t_idx = 0

        return successfull


    def bounce_kick(self, state, which_wall):
        """Kick the puck to the specified wall and bounce to adversary's goal."""
        puck_pos = state.get_puck_pos()
        p_goal = self.get_adversary_goal_pos()
        successful, v_puck_desired = self.contact_optimizer.bounce_pass_wall(puck_pos, p_goal, which_wall)
        if successful:
            p0 = state.get_player_pos(self.team, self.player_id)
            v0 = state.get_player_vel(self.team, self.player_id)
            p0_puck = state.get_puck_pos()
            v0_puck = state.get_puck_vel()
            successfull, self.u_traj = self.linear_optimizer.min_time_bounce_kick_traj(p0, v0, p0_puck, v0_puck, v_puck_desired)

            if successful:
                print("min_time_bounce_kick_traj optimization succeeded.")

            # Store trajectory and reset execution timer
        self.t_idx = 0

        return successfull



# TODO stays in front of the goal trying to intercept the ball
    def defend(self, state):

        p0 = state.get_player_pos(self.team, self.player_id)
        v0 = state.get_player_vel(self.team, self.player_id)

        pf_y = state.get_puck_pos()[1]
        defense_line = 0.3
        pf_x = 0.0
        if self.field > 0:
            pf_x = self.params.arena_limits_x/2.0 - defense_line
        else:
            pf_x = -self.params.arena_limits_x/2.0 + defense_line

        successfull, self.u_traj = self.linear_optimizer.min_time_traj(p0, v0, np.array([pf_x, pf_y]), np.zeros(2),  None)
        self.t_idx = 0

        return successfull

    # where the ball should be kicked
    def get_adversary_goal_pos(self):
        # TODO: define game parameter class and pass it around
        if self.field > 0:
            return np.array([-self.params.arena_limits_x/2.0, 0.0])
        else :
            return np.array([self.params.arena_limits_x/2.0, 0.0])

    def get_shoot_direction(self, p_goal, p_puck):
        """Returns direction to kick to reach goal"""
        shoot_direction = p_goal - p_puck
        if np.linalg.norm(shoot_direction) > 1e-4:
            shoot_direction/=np.linalg.norm(shoot_direction)
        return shoot_direction




