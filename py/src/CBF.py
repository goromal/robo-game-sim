import numpy as np
import matplotlib.pyplot as plt

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


    def get_centralized_safe_control(self, u_nominal, velocities, positions):
        prog = MathematicalProgram()

        N = len(velocities)

        u_to_solve = [prog.NewContinuousVariables(2, name='u_{}'.format(idx)) for idx in range(len(u_nominal))]

        for i in range(N):
            for j in range(N):
                if i is not j:
                    pij = positions[i] - positions[j]
                    vij = velocities[i] -velocities[j]

                    dxi = self.A_c.dot(np.concatenate((positions[i],velocities[i]), axis=0)) \
                          + self.B_c.dot(u_to_solve[i])

                    dxj = self.A_c.dot(np.concatenate((positions[j],velocities[j]), axis=0)) \
                          + self.B_c.dot(u_to_solve[j])

                    # if the robots are moving towards each other, add CBF constraint
                    if pij.dot(vij)/np.linalg.norm(pij) < 0:
                        # prog.AddConstraint(self.dhdx(pij, vij).dot(dxi) >= -self.alpha(self.h(pij, vij)))
                        # prog.AddConstraint(self.dhdx(pij, vij).dot(-dxj) >= -self.alpha(self.h(pij, vij)))

                        pij_norm = np.linalg.norm(pij)
                        vij_norm = np.linalg.norm(vij)
                        a_max = self.params.input_limit
                        delta_a_max = 2*a_max
                        hij = pij.dot(vij)/np.linalg.norm(pij) + np.sqrt(max(0, 2*delta_a_max*(np.linalg.norm(pij)-self.safety_radius))) # use max to sqrt of negative values
                        Bij = 1/hij

                        try:
                            prog.AddConstraint(-pij.dot(u_to_solve[i]-u_to_solve[j]) <=
                                               self.params.gamma/Bij * hij**2 *pij_norm -
                                               (vij.dot(pij))**2 / pij_norm**2 +
                                               vij_norm**2 + delta_a_max*vij.dot(pij) / (2*delta_a_max*(pij_norm-self.safety_radius)))
                        except:
                            print("Cannot add constraint..")


        # Add input limits
        for u in u_to_solve:
            prog.AddConstraint(u[0] <= self.params.input_limit)
            prog.AddConstraint(u[1] <= self.params.input_limit)
            prog.AddConstraint(u[0] >= -self.params.input_limit)
            prog.AddConstraint(u[1] >= -self.params.input_limit)


        # Minimally changing the nomial control
        u_diff = np.concatenate(u_nominal) - np.concatenate(u_to_solve)
        prog.AddCost(u_diff.dot(u_diff))

        result = Solve(prog)

        # TODO: get the new output..
        if result.is_success():
            print("Nominal controls: {}".format(u_nominal))
            print("Safe controls: {}".format(result.GetSolution(u_to_solve)))
            return result.GetSolution(u_to_solve)
        else:
            print("failed")
            return u_nominal

    def get_safe_control(self, u_nominal, velocities, positions, indices_to_solve):
        # velocities: list of every players velocities
        # positions: every players positions
        # indices_to_solve: indices of the players to return the safe control for, e.g., [0, 2]

        prog = MathematicalProgram()

        N = len(velocities)

        # TODO: initialize control variables to solve for
        u_to_solve = [prog.NewContinuousVariables(2, name='u_{}'.format(idx)) for idx in indices_to_solve]

        for i in indices_to_solve:
            for j in range(N):
                if j is not i:
                    pij = positions[i] - positions[j]
                    vij = velocities[i] -velocities[j]

                    dxi = self.A_c.dot(np.concatenate((positions[i],velocities[i]), axis=0))\
                          + self.B_c.dot(u_to_solve[i])

                    # if the robots are moving towards each other, add CBF constraint
                    if pij.dot(vij)/np.linalg.norm(pij) < 0:
                        prog.AddConstraint(self.dhdx(pij, vij).dot(dxi) >= -self.alpha(self.h(pij, vij)))

                        # a_max = np.abs(self.get_normalized_vector(pij).dot(self.get_normalized_vector(vij)) * self.params.input_limit)
                        # delta_a_max = 2*a_max
                        # hij = pij.dot(vij)/np.linalg.norm(pij) + np.sqrt(2*delta_a_max*(np.linalg.norm(pij)-self.safety_radius))
                        # Bij = 1/hij
                        # prog.AddConstraint(-pij.dot())


        # Add input limits
        for u in u_to_solve:
            prog.AddConstraint(u[0] <= self.params.input_limit)
            prog.AddConstraint(u[1] <= self.params.input_limit)
            prog.AddConstraint(u[0] >= -self.params.input_limit)
            prog.AddConstraint(u[1] >= -self.params.input_limit)


         # Minimally changing the nomial control
        u_diff = np.concatenate(u_nominal) - np.concatenate(u_to_solve)
        prog.AddCost(u_diff.dot(u_diff))

        result = Solve(prog)

        # TODO: get the new output..
        if result.is_success():
            print("Nominal controls: {}".format(u_nominal))
            print("Safe controls: {}".format(result.GetSolution(u_to_solve)))
            return result.GetSolution(u_to_solve[0]), result.GetSolution(u_to_solve[1])
        else:
            print("failed")
            return np.zeros(2), np.zeros(2)




    def h(self, pij, vij):
        v_normal = pij.dot(vij)/np.linalg.norm(pij)
        a_max = self.params.input_limit
        # a_max = np.abs(self.get_normalized_vector(pij).dot(self.get_normalized_vector(vij)) * self.params.input_limit)
        tau = self.params.tau_player
        Tb = tau * np.log(1+v_normal/(2*a_max))
        pij_norm = np.linalg.norm(pij)

        dist_when_stopped =  pij_norm + tau*(1-np.exp(-Tb/tau)) *v_normal \
                             + 2*a_max*(tau-tau*np.exp(-Tb/tau)-Tb)


        return dist_when_stopped - self.safety_radius

    def dhdx(self, pij, vij):
        v_normal = pij.dot(vij)/np.linalg.norm(pij)
        a_max = self.params.input_limit
        # a_max = np.abs(self.get_normalized_vector(pij).dot(self.get_normalized_vector(vij)) * self.params.input_limit)
        tau = self.params.tau_player
        Tb = tau * np.log(1+v_normal/(2*a_max))
        pij_norm = np.linalg.norm(pij)
        I = np.eye(2)

        dhdp = pij/pij_norm + tau*(1-np.exp(-Tb/tau)) * np.matmul(I/pij_norm - np.outer(pij, pij)/(pij_norm**3), vij)
        dhdv = tau*(1-np.exp(-Tb/tau))/pij_norm*pij

        return np.concatenate((dhdp, dhdv), axis=0)

    def alpha(self, h):
        # return self.barrier_gain * h
        return self.barrier_gain * h**3

    def get_normalized_vector(self, v):
        """Get normalized vector."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
