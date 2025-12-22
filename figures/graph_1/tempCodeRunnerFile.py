
# %%
fig, axs = plt.subplots(1, 2, figsize=(7, 3))  # width=7 in, height=3 in
i = 1
x = X[i]
y = Y[i]
z = Z[i]


plot_2d_heatmap(axs[0], x, y, z)


i = 5
x = X[i]
y = Y[i]
z = Z[i]

plot_2d_heatmap(axs[1], x, y, z)

plt.tight_layout()

plt.savefig("lorentzian.png", dpi=300)
plt.show()
