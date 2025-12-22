from matplotlib import pyplot as plt
from simi.utils import *
import matplotlib.gridspec as gridspec
from simi.run_simulation import *


def run_all(args=qubit_args):
    matrix = []

    amplitudes = np.linspace(0.0001, args["rabi_frequency"], args["rabi_points"])

    detunings = np.linspace(
        -args["detuning_span"] / 2,
        args["detuning_span"] / 2,
        args["detuning_points"],
    )
    for i, amp in enumerate(amplitudes):
        print(f"Amplitude = {i}/{len(amplitudes)}")
        args["rabi_frequency"] = amp
        vec = parallel_map(single_run_change_dict, detunings, task_args=(args,))

        if args["full_state"]:
            vec = np.array(vec)

        matrix.append(vec)
    return detunings, amplitudes, np.array(matrix)


def find_fwhm(x, y, z):
    FWHMs = []
    SNRs = []
    for i, amp in enumerate(x):
        vec = z[i]
        try:
            fwhm, snr = FWHM(x, np.array(vec), plot=True)
        except:
            fwhm = None
            snr = None
            print("no FWHM!")
        FWHMs.append(fwhm)
        SNRs.append(snr)

    return np.array(FWHMs), np.array(SNRs)


def plot_1d(ax: plt.Axes, x, z, **kwargs):
    ax.plot(x, z, **kwargs)
    # ax.set_xlabel("Detuning [MHz]")
    # ax.set_ylabel("Excited state population")
    # ax.set_ylim([0, 0.5])
    # ax.set_xlim([detunings[0] / 2 / pi / MHz, detunings[-1] / 2 / pi / MHz])
    # ax.axvline(x=-T2_limit / 2 / MHz, color="r", linestyle="--")
    # ax.axvline(x=T2_limit / 2 / MHz, color="r", linestyle="--")
    # ax.legend(loc="upper right")
    return ax


def plot_2d(ax: plt.Axes, x, y, z, **kwargs):

    # fwhms, snrs = find_fwhm(x, y, z)

    sc = ax.pcolormesh(x, y, z, **kwargs)
    # ax.plot(-fwhms[1:] / 2 / MHz / 2 / pi, amplitudes[1:] / 2 / pi / MHz, ".g")
    # ax.plot(fwhms[1:] / 2 / MHz / 2 / pi, amplitudes[1:] / 2 / pi / MHz, ".g")

    ax.axvline(x=-T2_limit / 2 / MHz, color="r", linestyle="--", label="T2")
    ax.axvline(x=T2_limit / 2 / MHz, color="r", linestyle="--", label="T2")

    ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Rabi Amplitude [MHz]")
    # ax.set_ylim([0, amplitudes[-1] / 2 / pi / MHz])
    # ax.set_xlim([detunings[0] / 2 / pi / MHz, detunings[-1] / 2 / pi / MHz])

    return ax, sc


def find_peak(ax, x, y, z):
    bound = 0.05
    mask = (x >= -bound) & (x <= bound)
    z_masked = z[:, mask]
    x_masked = x[mask]

    min_indices = np.argmin(z_masked, axis=1)
    x_min = x_masked[min_indices]

    # x_min = np.array([x_data[np.argmin(z,bounds=[0, 1])] for z in z_data])
    ax.plot(x_min, y, "r.-", label="Min across amplitudes")

    max_idx = np.argmax(y)
    return x_min, y


if __name__ == "__main__":
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 8.5), gridspec_kw={"height_ratios": [1, 2]}
    )

    qubit_args["cutoff"] = 1e-3
    qubit_args["rabi_points"] = 11
    qubit_args["full_state"] = True

    detunings, amplitudes, z = run_all(args=qubit_args)

    print(z.shape)
    x = detunings / 2 / pi / MHz
#     y = amplitudes / 2 / pi / MHz
#     # fwhms, snrs = find_fwhm(x, y, z)

#     ax1, ax2 = axes

#     #     for zi in z:
#     #         plot_1d(ax1, x, zi)
#     _, с = plot_2d(ax2, x, y, z, shading="auto")

#     #     # fig.colorbar(с, ax=ax2, label="Excited state population")
#     #     # x_min, _ = find_peak(ax2, x, y, z)

#     #     # ax1.set_title(
#     #     #     f'pulse type is {qubit_args["pulse_type"]}, eco is {qubit_args["eco_pulse"]} \n pulse length = {qubit_args["pulse_length"] / us:0.3f} us \n cutoff = {qubit_args["cutoff"]} \n $T_1$ = {T1 / us:.2f} us, $T_d$ = {T_dephasing / us:.2f} us -> $T_2$ = {T2 / us:.1f} us'
#     #     # )
#     #     # fig.tight_layout()

#     plt.show()

# # # %%
