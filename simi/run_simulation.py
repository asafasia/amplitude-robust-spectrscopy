from simi.utils import *
from simi.Hamiltonian_helper import Hamiltonian

c_ops = [
    np.sqrt(1 / qubit_args["T1"]) * a,
    np.sqrt(2 / qubit_args["T_dephasing"]) * a.dag() * a,
]


def single_run_change_dict(detuning, args):
    args["detuning"] = detuning
    tlist = np.linspace(-args["pulse_length"] / 2, args["pulse_length"] / 2, 1000)
    H = Hamiltonian(
        rotated_frame=True,
        anharmonicity=args["anharmonicity"],
        detuning=detuning,
        rabi_frequency=args["rabi_frequency"],
        pulse_type=args["pulse_type"],
        eco_pulse=args["eco_pulse"],
    ).get_hamiltonian()
    state = basis(N_dim, 0)

    # if kawargs is key full_state=True
    # return the full state
    if args.get("full_state", True):
        expect_result = mesolve(
            H, state, tlist, c_ops, e_ops=[sigmax(), sigmay(), sigmaz()], args=args
        ).expect

        final_state = [
            expect_result[0][-1],
            expect_result[1][-1],
            expect_result[2][-1],
        ]
        return final_state

    else:
        return mesolve(H, state, tlist, c_ops, e_ops=[a.dag() * a], args=args).expect[
            0
        ][-1]


def double_run_change_dict(detuning, change_dict=None):
    psi0 = (basis(N_dim, 0) + 1 * basis(N_dim, 1)).unit()
    psi1 = (basis(N_dim, 0) - 1 * basis(N_dim, 1)).unit()
    sim_args = {
        "w": qubit_args["qubit_frequency"] - detuning,
        "anharmonicity": qubit_args["anharmonicity"],
        "rabi_frequency": qubit_args["rabi_frequency"],
        "cutoff": qubit_args["cutoff"],
        "pulse_length": qubit_args["pulse_length"],
        "zeroed_pulse": qubit_args["zeroed_pulse"],
        "n": qubit_args["n"],
        "eco_pulse": qubit_args["eco_pulse"],
    }
    if change_dict:
        sim_args.update(change_dict)

    tlist = np.linspace(
        -sim_args["pulse_length"] / 2, sim_args["pulse_length"] / 2, 1000
    )

    H = Hamiltonian(
        rotated_frame=True,
        anharmonicity=sim_args["anharmonicity"],
        detuning=detuning,
        rabi_frequency=sim_args["rabi_frequency"],
        pulse_type=qubit_args["pulse_type"],
        eco_pulse=sim_args["eco_pulse"],
    ).get_hamiltonian()

    vec1 = mesolve(H, psi0, tlist, c_ops, e_ops=[a.dag() * a], args=sim_args).expect[0][
        -1
    ]
    vec2 = mesolve(H, psi1, tlist, c_ops, [a.dag() * a], args=sim_args).expect[0][-1]
    return abs(vec1 - vec2) / 2


if __name__ == "__main__":
    detuning = 0 * 2 * pi * MHz

    qubit_args["full_state"] = True
    # print(single_run_change_dict(detuning, qubit_args))
