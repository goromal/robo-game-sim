import numpy as np
from src.DMPC import DMPC

class DMPCPlayer(object):
    # play types
    OFFENSE = 0
    DEFENSE = 1
    # player types
    ATTACKER = DMPC.ATTACKER # plays more offense on average
    DEFENDER = DMPC.DEFENDER # plays more defense on average

    def __init__(self, sim_params, controller_params, strategy_params, field, player_id):
        # player parameters
        self.params = sim_params
        self.strategy_params = strategy_params
        # field tells which side of the arena the player defends -1 left, +1 right
        self.home_pos =  field * np.array([self.params.arena_limits_x/2.0, 0.0])
        self.goal_pos = -field * np.array([self.params.arena_limits_x/2.0, 0.0])
        if field < 0: self.this_team = "A"
        else: self.this_team = "B"
        self.player_id = player_id # 1 or 2; 1 plays more offense, 2 plays more defense
        self.field = field

        # controller
        self.controller = DMPC(sim_params, controller_params)

    def get_action(self, play, state):
        x_des = np.zeros((4,1))
        puck_pos = state.get_puck_pos()

        if   play == DMPCPlayer.OFFENSE:
            if   self.player_id == DMPCPlayer.ATTACKER:
                # Get to puck wherever it is and hit it towards the goal!
                hit_dir = self.goal_pos - puck_pos
                hit_dir = self.strategy_params.v_hit * hit_dir / np.linalg.norm(hit_dir)
                x_des = np.array([puck_pos[0], puck_pos[1], hit_dir[0], hit_dir[1]])
            elif self.player_id == DMPCPlayer.DEFENDER:
                # If puck is within range, hit towards goal! Else defend.
                if self.field * puck_pos[0] > 0:
                    def_pos = self.home_pos + (puck_pos - self.home_pos) / 2.0
                    x_des = np.array([def_pos[0], def_pos[1], 0., 0.])
                else:
                    hit_dir = self.goal_pos - puck_pos
                    hit_dir = self.strategy_params.v_hit * hit_dir / np.linalg.norm(hit_dir)
                    x_des = np.array([puck_pos[0], puck_pos[1], hit_dir[0], hit_dir[1]])
        elif play == DMPCPlayer.DEFENSE:
            if   self.player_id == DMPCPlayer.ATTACKER:
                # Deflect puck trajectory in any way possible to return to offense
                hit_dir = self.goal_pos - puck_pos
                hit_dir = 2 * self.strategy_params.v_hit * hit_dir / np.linalg.norm(hit_dir)
                x_des = np.array([puck_pos[0], puck_pos[1], hit_dir[0], hit_dir[1]])
            elif self.player_id == DMPCPlayer.DEFENDER:
                # Get between the puck and the home goal
                def_pos = self.home_pos + (puck_pos - self.home_pos) / 2.0
                x_des = np.array([def_pos[0], def_pos[1], 0., 0.])

        return self.controller.compute_control(x_des, state, self.this_team, self.player_id)
