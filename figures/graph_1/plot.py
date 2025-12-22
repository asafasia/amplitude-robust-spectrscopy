import os
import json

from matplotlib import pyplot as plt
import numpy as np


def extract_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


data_folder_path = "data"

files = [file for file in os.listdir(data_folder_path) if file.endswith(".json")]

for file in files:
    print(file)

# for file in files:
#     file_path = os.path.join(data_folder_path, file)
#     with open(file_path, "r") as f:
#         data = f.read()
#         print(f"Contents of {file}:\n{data}\n")


# files = [
#     "lorentzian-echo-True-2d-40.0-us_0.0001.json",
#     "lorentzian-echo-False-2d-40.0-us_0.0001.json",
# ]
X = []
Y = []
Z = []

for file in files:
    file_path = os.path.join(data_folder_path, file)
    data = extract_data_from_json(file_path)
    measured_data = data["measured_data"]
    sweep_parameters = data["sweep_parameters"]
    states = measured_data["states"]
    detunigs = sweep_parameters["detuning"]
    amplitudes = sweep_parameters["amplitudes"]
    X.append(np.array(detunigs))
    Y.append(np.array(amplitudes))
    Z.append(np.array(states))


def plot_2d_heatmap(ax, x, y, z):
    c = ax.pcolormesh(x / 1e6, y, z, shading="auto")


# %%
fig, axs = plt.subplots(1, 2, figsize=(7, 3))  # width=7 in, height=3 in
i = 1
x = X[i]
y = Y[i]
z = Z[i]


# plot_2d_heatmap(axs[0], x, y, z)


# i = 5
# x = X[i]
# y = Y[i]
# z = Z[i]

# plot_2d_heatmap(axs[1], x, y, z)

# plt.tight_layout()

# plt.savefig("lorentzian.png", dpi=300)
# plt.show()


# %%

for i in range(len(files)):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))  # width=7 in, height=3 in
    plot_2d_heatmap(ax, X[i], Y[i], Z[i])
    plt.show()

# from scipy import signal

# for i, zi in enumerate(z):
#     z[i] = signal.savgol_filter(z[i], window_length=3, polyorder=2)

# z = z.T

# for i, zi in enumerate(z):
#     z[i] = signal.savgol_filter(z[i], window_length=3, polyorder=2)

# z = z.T

# # c = ax.pcolormes /h(x / 1e6, y, z, shading="auto")

# # fig.colorbar(c, ax=ax, label="States")
# # ax.set_xlabel("Detuning")
# # ax.set_ylabel("Amplitudes")
# # ax.set_title("Lorentzian Echo True")
# # plt.show()
# ax.plot(x / 1e6, z[10])
# ax.set_xlabel("Detuning (MHz)")
# ax.set_ylabel("States")
# ax.set_title("Lorentzian Echo True")
# plt.show()

# %%
