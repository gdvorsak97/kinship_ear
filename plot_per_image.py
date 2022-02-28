from matplotlib import pyplot as plt
import numpy as np
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

md_count = 7305
ms_count = 7500
fd_count = 7281
fs_count = 7443
bs_count = 5109
bb_count = 1332
ss_count = 1312

# col = [colors[0], colors[1], colors[2], colors[3], colors[4], colors[6]]
names = ["M-D", "M-So", "F-D", "F-So", "B-Si", "B-B", "Si-Si"]
data = [md_count, ms_count, fd_count, fs_count, bs_count, bb_count, ss_count]
fig = plt.figure()
plt.rcParams.update({'font.size': 18})
ax = fig.add_subplot(111)
ax.set_yticks(np.arange(0, 8000, 1000))
ax.bar(names, data)
plt.grid(axis='y')
plt.title("Number of per-image relations")
destination = "C:\\Users\\Grega\\Desktop\\output\\"
filename = destination + "chart_imgs.pdf"
plt.savefig(filename, bbox_inches='tight')
plt.show()
