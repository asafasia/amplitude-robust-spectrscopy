import time
from matplotlib import pyplot as plt
from utils import *
import matplotlib.gridspec as gridspec
from run_simulation import *


def run_all(args=qubit_args):
    matrix = []
    for i, amp in enumerate(amplitudes):
        print(f"Amplitude = {i}/{len(amplitudes)}")
        args["rabi_frequency"] = amp
        vec = parallel_map(single_run_change_dict, detunings, task_args=(args,))
        matrix.append(vec)
    return np.array(matrix)


def find_fwhm(x, y, z):
    FWHMs = []
    SNRs = []
    for i, amp in enumerate(amplitudes):
        vec = z[i]
        try:
            fwhm, snr = FWHM(detunings, np.array(vec))
        except:
            fwhm = None
            snr = None
            print("no FWHM!")
        FWHMs.append(fwhm)
        SNRs.append(snr)

    return np.array(FWHMs), np.array(SNRs)


def plot_1d(ax: plt.Axes, x, z, **kwargs):
    print(x.shape, z.shape)
    ax.plot(x, z, **kwargs)
    # ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Excited state population")
    # ax.set_ylim([0, 0.5])
    # ax.set_xlim([detunings[0] / 2 / pi / MHz, detunings[-1] / 2 / pi / MHz])
    ax.axvline(x=-T2_limit / 2 / MHz, color="r", linestyle="--")
    ax.axvline(x=T2_limit / 2 / MHz, color="r", linestyle="--")
    ax.legend(loc="upper right")
    return ax


def plot_2d(ax: plt.Axes, x, y, z, **kwargs):

    fwhms, snrs = find_fwhm(x, y, z)

    sc = ax.pcolormesh(x, y, z, **kwargs)
    ax.plot(-fwhms[1:] / 2 / MHz / 2 / pi, amplitudes[1:] / 2 / pi / MHz, ".g")
    ax.plot(fwhms[1:] / 2 / MHz / 2 / pi, amplitudes[1:] / 2 / pi / MHz, ".g")

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


lengths = [10 * us, 20 * us, 30 * us, 40 * us, 50 * us, 60 * us]

for l in lengths:
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 8.5), gridspec_kw={"height_ratios": [1, 2]}
    )
    ax1, ax2 = axes
    print(f"pulse length = {l/us} us")
    qubit_args["pulse_length"] = l
    z = run_all(qubit_args)
    x = detunings / 2 / pi / MHz
    y = amplitudes / 2 / pi / MHz
    fwhms, snrs = find_fwhm(x, y, z)
    x_min, _ = find_peak(ax2, x, y, z)

    for zi in z:
        plot_1d(ax1, x, zi)
    plot_2d(ax2, x, y, z)

    ax1.set_title(
        f'pulse type is {qubit_args['pulse_type']}, eco is {qubit_args['eco_pulse']} \n pulse length = {qubit_args["pulse_length"] / us:0.3f} us \n cutoff = {qubit_args["cutoff"]} \n $T_1$ = {T1 / us:.2f} us, $T_d$ = {T_dephasing / us:.2f} us -> $T_2$ = {T2 / us:.1f} us'
    )
    fig.tight_layout()

    def save():
        if N_dim == 2:
            two_level = "two_level"
        elif N_dim == 3:
            two_level = "three_level"
        folder_path = f"graphs/data_sim/{two_level}/"
        exp_name = f'lorentzian-echo-{qubit_args["eco_pulse"]}-2d-{qubit_args["pulse_length"] / us:.0f}-us-{qubit_args["cutoff"]}'
        sweep_parms = {
            "amplitudes": amplitudes.tolist(),
            "detunings": detunings.tolist(),
        }
        measured_data = {"states": z.tolist()}

        analysis = {
            "x_min": x_min.tolist(),
            "fwhms": fwhms.tolist(),
            "snrs": snrs.tolist(),
        }

        data = {
            "sweep_parms": sweep_parms,
            "measured_data": measured_data,
            "analysis": analysis,
        }

        import json

        # save file

        file_path = f"{folder_path}{exp_name}.json"
        print(file_path)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    save()
