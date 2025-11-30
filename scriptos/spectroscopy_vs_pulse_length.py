# %%
# import time
from matplotlib import pyplot as plt
from simi.utils import *
import matplotlib.gridspec as gridspec
from simi.run_simulation import *


def run_all(args):
    matrix = []

    lengths = np.linspace(
        0.1*us, args["pulse_length"], args["length_points"]
    )

    detunings = np.linspace(
        -args["detuning_span"] / 2,
        args["detuning_span"] / 2,
        args["detuning_points"],
    )
    for i, length in enumerate(lengths):
        print(f"Length = {i}/{len(lengths)}")
        args["pulse_length"] = length
        vec = parallel_map(single_run_change_dict,
                           detunings, task_args=(args,))
        matrix.append(vec)
    return detunings, lengths,   np.array(matrix)


def find_fwhm(x, y, z):
    FWHMs = []
    SNRs = []
    for i, l in enumerate(y):
        vec = z[i]
        try:
            fwhm, snr = FWHM(x, np.array(vec))
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
    ax.set_ylabel("Excited state population")
    # ax.set_ylim([0, 0.5])
    # ax.set_xlim([detunings[0] / 2 / pi / MHz, detunings[-1] / 2 / pi / MHz])
    # ax.axvline(x=-T2_limit / 2 / MHz, color="r", linestyle="--")
    # ax.axvline(x=T2_limit / 2 / MHz, color="r", linestyle="--")
    # ax.legend(loc="upper right")
    return ax


def plot_2d(ax: plt.Axes, x, y, z, **kwargs):

    # fwhms, snrs = find_fwhm(x, y, z)

    sc = ax.pcolormesh(x, y, z, **kwargs, vmin=0, vmax=0.6)
    # ax.plot(-fwhms[1:] / 2 / MHz / 2 / pi, lengths[1:] / 2 / pi / MHz, ".g")
    # ax.plot(fwhms[1:] / 2 / MHz / 2 / pi, lengths[1:] / 2 / pi / MHz, ".g")

    ax.axvline(x=-T2_limit / 2 / MHz, color="r", linestyle="--", label="T2")
    ax.axvline(x=T2_limit / 2 / MHz, color="r", linestyle="--", label="T2")

    ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Pulse Length [us]")
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


def return_fwhms(amp, length_points=21, pulse_length=100*us):
    qubit_args['cutoff'] = 0.99999
    qubit_args['eco_pulse'] = False
    qubit_args['length_points'] = length_points
    qubit_args['rabi_frequency'] = amp
    qubit_args['pulse_length'] = pulse_length
    qubit_args['detuning_span'] = 0.5 *2 * pi * MHz if amp < 0.1*MHz*2*pi else amp*3

    x, lengths, z = run_all(qubit_args)

    fwhms, snrs = [], []

    for zi in z:
        mu, fwhm, snr = FWHM(x, zi, plot=False)
        fwhms.append(fwhm)
        snrs.append(snr)

    return np.array(lengths[1:]), np.array(fwhms[1:]), np.array(snrs[1:])


# %%
if __name__ == "__main__":

    qubit_args['cutoff'] = 0.99999
    qubit_args['eco_pulse'] = False
    qubit_args['rabi_frequency'] = 2 * pi * 0.002 * MHz

    detunings, lengths, z = run_all(qubit_args)


# %%
    x = detunings
    y = lengths
    fwhms, snrs = [], []

    for zi in z:
        mu, fwhm, snr = FWHM(x, zi, plot=False)
        fwhms.append(fwhm)
        snrs.append(snr)

    fwhms = np.array(fwhms)
    snrs = np.array(snrs)
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 8.5), gridspec_kw={"height_ratios": [1, 2]}
    )
    ax1, ax2 = axes

    indexs = [1]
    for i in indexs:
        # FWHM(x, z[i],plot=True)
        plot_1d(ax1, x, z[i], label=f"Pulse length = {lengths[i]/us:0.2f} us")

    _, c = plot_2d(ax2, x/MHz/2/pi, y/us, z)

    fig.colorbar(c, ax=ax2, label="Excited state population", pad=0.04)
    x_min, _ = find_peak(ax2, x, y, z)

    ax2.plot(-fwhms[1:] / MHz/2/pi/2, lengths[1:] /
             us, label="FWHM", color='green')
    ax2.plot(fwhms[1:] / MHz/2/pi / 2, lengths[1:] /
             us, label="FWHM", color='green')

    ax1.set_title(
        f'pulse type is {qubit_args["pulse_type"]}, eco is {qubit_args["eco_pulse"]} \n pulse length = {qubit_args["pulse_length"] / us:0.3f} us \n cutoff = {qubit_args["cutoff"]} \n $T_1$ = {T1 / us:.2f} us, $T_d$ = {T_dephasing / us:.2f} us -> $T_2$ = {T2 / us:.1f} us'
    )
    ax2.set_xlim([detunings[0] / 2 / pi / MHz, detunings[-1] / 2 / pi / MHz])
    fig.tight_layout()
    plt.show()

# %%


# %%
