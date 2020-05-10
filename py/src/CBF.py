import numpy as np
from scipy.special import comb

# pydrake imports
from pydrake.all import eq, MathematicalProgram, Solve, Variable, LinearSystem, DirectTranscription, DirectCollocation, PiecewisePolynomial, SnoptSolver

class CBF:
    def __init__(self, params, safety_radius, barrier_gain):
        # initialize game parameters and system dyanmics
        self.params = params
        self.A_c = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_player, 0], [0, 0, 0, -1.0/self.params.tau_player]])
        self.B_c = np.array([[0, 0], [0, 0], [1.0/self.params.tau_player, 0], [0, 1.0/self.params.tau_player]])
        self.C = np.eye(4)
        self.D = np.zeros((4,2))

        # initialize CBF related params
        assert(safety_radius > self.params.player_radius)
        assert(barrier_gain > 0)
        self.safety_radius = safety_radius
        self.barrier_gain = barrier_gain # lower = slower

    def get_safe_control(self, u_nominal, velocities, positions, indices_to_solve):
        # velocities: list of every players velocities
        # positions: every players positions
        # indices_to_solve: indices of the players to return the safe control for, e.g., [0, 2]

        prog = MathematicalProgram()

        N = len(velocities)
        num_constraints = int(comb(N, 2))  # number of pair-wise collision avoidance constraints

        # TODO: initialize control variables to solve for
        u_to_solve = [prog.NewContinuousVariables(2, name='u_{}'.format(idx)) for idx in indices_to_solve]

        for i in u_to_solve:
            for j in range(N):
                if j is not i:
                    pi = positions[i]
                    pj = positions[j]
                    vi = velocities[i]
                    vj = velocities[j]

                    pij =  pi - pj
                    vij = vi -vj

                    dxi = self.A_c.dot(np.concatenate((pi, vi), axis=0)) + self.B_c.dot(u_to_solve[i])
                    prog.AddConstraint(self.dhdx(pij, vij).dot(dxi) >= -self.alpha(self.h(pij, vij)))

                    # dxj = self.A_c.dot(np.concatenate((pj, vj), axis=0)) + self.B_c.dot(u_to_solve[j])
                    # prog.AddConstraint(self.dhdx(pij, vij).dot(-dxj) >= -self.alpha(self.h(pij, vij)))


         # Minimally change the control
        for i, u_nom in enumerate(u_nominal):
            prog.AddCost((u_nom-u_to_solve[i]).dot(u_nom-u_to_solve[i]))

        result = Solve(prog)

        # TODO: get the new output..





    def h(self, pij, vij):
        v_normal = pij.dot(vij)/np.linalg.norm(pij)
        a_max = self.params.input_limit
        tau = self.params.tau_player
        Tb = tau * np.log(1+v_normal/(2*a_max))
        pij_norm = np.linalg.norm(pij)

        dist_when_stopped =  pij_norm + tau*(1-np.exp(-Tb/tau)) / pij_norm * pij.dot(vij)\
                             + 2*a_max*(tau-tau*np.exp(-Tb/tau)-Tb)

        return dist_when_stopped - self.safety_radius

    def dhdx(self, pij, vij):
        v_normal = pij.dot(vij)/np.linalg.norm(pij)
        a_max = self.params.input_limit
        tau = self.params.tau_player
        Tb = tau * np.log(1+v_normal/(2*a_max))
        pij_norm = np.linalg.norm(pij)
        I = np.eye(2)

        dhdp = pij/pij_norm + tau*(1-np.exp(-Tb/tau)) * np.matmul(vij, I/pij_norm - np.outer(pij, pij)/(pij_norm**3))
        dhdv = tau*(1-np.exp(-Tb/tau))/pij_norm*pij

        return np.concatenate((dhdp, dhdv), axis=0)

    def alpha(self, h):
        return self.barrier_gain * h**3