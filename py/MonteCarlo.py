from robo_game_py import GameSim
from src.SimState import SimState
import numpy as np
from math import ceil
from array import array
import os, sys, progressbar

def monte_carlo(mc_params, team_A_run_method, team_B_run_method, cbf_run_method=None):
    sim = GameSim()

    if not os.path.exists(mc_params.log_prefix):
        os.makedirs(mc_params.log_prefix)

    output_file = open(os.path.join(mc_params.log_prefix, 'configuration.txt'), 'w')
    param_array = [mc_params.num_runs, mc_params.T, mc_params.dt, mc_params.winning_score,
                   mc_params.x0_ball[0], mc_params.x0_ball[1], mc_params.x0_ball[2],
                   mc_params.x0_ball[3], mc_params.noise_stdev, mc_params.tau_puck,
                   mc_params.tau_player, mc_params.player_mass, mc_params.puck_mass]
    for param in param_array:
        output_file.write(str(param) + '\n')
    output_file.close()

    for i in range(1, mc_params.num_runs + 1):
        print('MC RUN # {} / {}:'.format(i, mc_params.num_runs))

        logname = os.path.join(mc_params.log_prefix, 'mc_run_{}.log'.format(i))

        sim.reset(mc_params.dt, mc_params.winning_score, mc_params.x0_ball, mc_params.noise_stdev,
                  True, logname, i, mc_params.tau_puck, mc_params.tau_player,
                  mc_params.player_mass, mc_params.puck_mass)

        t = 0.0
        sim_state = SimState(sim.run(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)))
        N = int(ceil(mc_params.T / mc_params.dt))

        for i in progressbar.progressbar(range(N)):

            velA1, velA2 = team_A_run_method(sim_state)
            velB1, velB2 = team_B_run_method(sim_state)

            if not cbf_run_method is None:
                u_nominal = [velA1, velA2, velB1, velB2]
                velA1, velA2, velB1, velB2 = cbf_run_method(u_nominal, sim_state)

            sim_state = SimState(sim.run(velA1, velA2, velB1, velB2))

        print()

    print('DONE!')
