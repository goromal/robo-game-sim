import numpy as np
import matplotlib.pyplot as plt

# pydrake imports
from pydrake.all import eq, MathematicalProgram, Solve, Variable, LinearSystem, DirectTranscription, DirectCollocation, PiecewisePolynomial, SnoptSolver

class CBF:
    """
    Collision avoidance using centralized control barrier functions. Control inputs are minimally modified such that
    robots are guaranteed to avoid collisions with each other.

    The following system dynamics have been incorporated into barrier functions, so do not appear individually:
        self.A_c = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0/self.params.tau_player, 0], [0, 0, 0, -1.0/self.params.tau_player]])
        self.B_c = np.array([[0, 0], [0, 0], [1.0/self.params.tau_player, 0], [0, 1.0/self.params.tau_player]])
        self.C = np.eye(4)
        self.D = np.zeros((4,2))

    Parameters:
        barrier_gain > 0. The lower it is, the slower each player is allowed to approach each other.
        safety_radius > 0. The safety radius for each player, e.g., 2 * player_radius.
    """

    def __init__(self, params, safety_radius, barrier_gain):

        # initialize game parameters and system dyanmics
        self.params = params

        # initialize CBF related params
        assert safety_radius >= 2*self.params.player_radius, "Safety radius must be greater than twice the player radius."
        assert barrier_gain > 0, "Barrier gain must be greater than 0."
        self.safety_radius = safety_radius
        self.barrier_gain = barrier_gain


    def get_centralized_safe_control_damped_double_integrator(self, u_nominal, sim_state, debug=False):
        """Minimally modify the nomial control inputs, such that robots are guaranteed to avoid collisions."""

        positions, velocities = self.get_all_positions_and_velocities(sim_state)
        N = len(positions)

        prog = MathematicalProgram()
        u_to_solve = [prog.NewContinuousVariables(2, name='u_{}'.format(idx)) for idx in range(N)]

        # check pair-wise collisions
        for i in range(N):
            for j in range(N):
                if j is not i:

                    pij = positions[i] - positions[j]           # relative position
                    vij = velocities[i] -velocities[j]          # relative velocity
                    pij_norm = np.linalg.norm(pij)              # norm of relative position
                    pij_unit = self.get_normalized_vector(pij)  # normalized relative position
                    vcol = pij_unit.dot(vij)                    # the relative velocity that leads to collision

                    # if the robots are moving towards each other, add CBF constraint
                    if vcol < 0:

                        tau = self.params.tau_player
                        a_rel_max = 2 * self.params.input_limit
                        Ds = self.params.safety_radius

                        uij = u_to_solve[i] - u_to_solve[j]

                        # compute hij_dot
                        exp_factor = np.exp((pij_norm + tau*vcol-Ds)/(tau*a_rel_max))
                        vij2_m_vcol2 = vij.dot(vij)-vcol**2
                        hij_dot = exp_factor * ( -vij2_m_vcol2/pij_norm +
                                                 (a_rel_max-vcol)/(tau*a_rel_max)*(vcol+tau/pij_norm*vij2_m_vcol2)  +
                                                 vcol/(tau*a_rel_max)*(vcol-pij_unit.dot(uij)))

                        # compute hij
                        hij = (a_rel_max-vcol) * exp_factor - a_rel_max

                        # add CBF constraint that is linear in uij
                        prog.AddConstraint(hij_dot >= -self.alpha(hij))

        # add input limits
        self.add_input_limits(prog, u_to_solve)

        # minimally changing the nomial control
        u_diff = np.concatenate(u_nominal) - np.concatenate(u_to_solve)
        prog.AddCost(u_diff.dot(u_diff))

        # solve the QP and return the safe controls
        result = Solve(prog)

        if result.is_success():
            if debug:
                print("Nominal controls: {}".format(u_nominal))
                print("Safe controls: {}".format(result.GetSolution(u_to_solve)))
            return result.GetSolution(u_to_solve)
        else:
            print("CBF collision avoidance failed. Return nomial inputs.")
            return u_nominal

    def get_centralized_safe_control_double_integrator(self, u_nominal, sim_state, debug=False):
        """DO NOT USE.
        CBF formulation for double integrators based on https://ames.gatech.edu/ADHS15_Swarm_Barrier.pdf.
        However, this does not apply for our dynamics. """

        positions, velocities = self.get_all_positions_and_velocities(sim_state)
        N = len(positions)

        prog = MathematicalProgram()
        u_to_solve = [prog.NewContinuousVariables(2, name='u_{}'.format(idx)) for idx in range(len(u_nominal))]

        # check pair-wise collisions
        for i in range(N):
            for j in range(N):
                if i is not j:
                    pij = positions[i] - positions[j]
                    vij = velocities[i] -velocities[j]

                    # if the robots are moving towards each other, add CBF constraint
                    if pij.dot(vij)/np.linalg.norm(pij) < 0:

                        pij_norm = np.linalg.norm(pij)
                        vij_norm = np.linalg.norm(vij)
                        a_max = self.params.input_limit
                        delta_a_max = 2*a_max

                        # use max to sqrt of negative values
                        hij = pij.dot(vij)/np.linalg.norm(pij) + \
                              np.sqrt(max(0, 2*delta_a_max*(np.linalg.norm(pij)-self.safety_radius)))
                        Bij = 1/hij

                        try:
                            # add CBF constraint that is linear in uij
                            prog.AddConstraint(-pij.dot(u_to_solve[i]-u_to_solve[j]) <=
                                               self.params.gamma/Bij * hij**2 *pij_norm -
                                               (vij.dot(pij))**2 / pij_norm**2 +
                                               vij_norm**2 + delta_a_max*vij.dot(pij) / (2*delta_a_max*(pij_norm-self.safety_radius)))
                        except:
                            print("Cannot add constraint..")


        # add input limits
        self.add_input_limits(prog, u_to_solve)

        # minimally changing the nomial control
        u_diff = np.concatenate(u_nominal) - np.concatenate(u_to_solve)
        prog.AddCost(u_diff.dot(u_diff))

        # solve the QP and return the safe controls
        result = Solve(prog)

        if result.is_success():
            if debug:
                print("Nominal controls: {}".format(u_nominal))
                print("Safe controls: {}".format(result.GetSolution(u_to_solve)))
            return result.GetSolution(u_to_solve)
        else:
            print("CBF collision avoidance failed. Return nomial inputs.")
            return u_nominal

    def get_all_positions_and_velocities(self, sim_state):
        """Return all players' positions and velocities."""
        positions = [sim_state.get_player_pos("A", 1),
                     sim_state.get_player_pos("A", 2),
                     sim_state.get_player_pos("B", 1),
                     sim_state.get_player_pos("B", 2)
                     ]
        velocities = [sim_state.get_player_vel("A", 1),
                      sim_state.get_player_vel("A", 2),
                      sim_state.get_player_vel("B", 1),
                      sim_state.get_player_vel("B", 2)
                      ]
        return positions, velocities

    def alpha(self, h):
        """Extended class-K infinity functions. For instance, alpha(h)=h, alpha(h)=h**3."""
        return self.barrier_gain * h**3

    def add_input_limits(self, prog, all_inputs):
        """Add input limits to every input."""
        for u in all_inputs:
            prog.AddConstraint(u[0] <= self.params.input_limit)
            prog.AddConstraint(u[1] <= self.params.input_limit)
            prog.AddConstraint(u[0] >= -self.params.input_limit)
            prog.AddConstraint(u[1] >= -self.params.input_limit)

    def get_normalized_vector(self, v):
        """Get normalized vector."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
