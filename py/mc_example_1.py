from MonteCarlo import monte_carlo # !! REQUIRES progressbar2 Python 3 module
import numpy as np

# Imports for this specific experiment
from src.ClassicalTeam import ClassicalTeam
from src.CBF import CBF

# Parameter definition
class GameParams:
    def __init__(self):
        # Monte Carlo Parameters
        self.log_prefix = "classical_vs_classical"
        self.num_runs = 4
        self.noise_stdev = 0.0 # noise on input

        # Team Parameters
        self.T = 10.0 # seconds, max total time
        self.dt = 0.05
        self.player_radius = 0.2
        self.puck_radius = 0.175
        self.tau_player=0.5
        self.tau_puck = 0.1 # set to 1.0 for bounce_kick to work
        self.player_mass = 1.0
        self.puck_mass = 0.5
        self.input_limit= 10.0
        self.arena_limits_x=10.0
        self.arena_limits_y=5.0
        self.goal_height = 1.0
        self.winning_score = 4
        self.x0_ball = np.array([-1, 0.,0.,0.]) # np.array([0.,0.,0.,0.])
        self.gamma = 1.

        # CBF parameters.
        self.safety_radius = 2 * self.player_radius
        self.barrier_gain = 30  # the higher, the faster robots can approach each other
params = GameParams()

# Team/Method instantiation
home_team = ClassicalTeam(params, -1, "A") # team A defend left goal
away_team = ClassicalTeam(params, 1, "B")  # team B defend right goal
centralized_CBF = CBF(params, params.safety_radius, params.barrier_gain)

# Run simulations
monte_carlo(params, home_team.run, away_team.run, centralized_CBF.get_centralized_safe_control_damped_double_integrator)
