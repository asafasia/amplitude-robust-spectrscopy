# from utils import *
# from Hamiltonian_helper import Hamiltonian
#
# class Solver:
#     def __init__(self, hamiltonian):
#         self.hamiltonian = hamiltonian
#         self.c_opts = [
#             np.sqrt(1 / qubit_args["T1"]) * destroy(N_dim),
#             np.sqrt(1 / qubit_args["T_dephasing"]) * sigmaz() / np.sqrt(2)
#         ]
#
#     def solve(self, data, detuning, amplitude):
#         psi0 = basis(2, 0)
#
#         sim_args = {
#             'w': qubit_args['qubit_frequency'],
#             'cutoff': qubit_args['cutoff'],
#             'pulse_length': qubit_args['pulse_length'],
#             'zeroed_pulse': qubit_args['zeroed_pulse']
#         }
#
#         results = mesolve(self.hamiltonian, psi0, tlist, self.c_opts, [a.dag() * a], args=sim_args).expect[0][-1]
#
#         return results
