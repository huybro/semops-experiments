import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})
tuples = [10, 30, 50, 70, 90, 110, 130]

sem_map = [3.9900, 11.7673, 19.5892, 27.4974, 35.3807, 43.2082, 51.0603]#, 58.9298]
sem_filter = [0.4371, 1.1674, 1.9009, 27.2613, 35.9739, 43.9193, 51.8613]#, 59.8752]

throughput = [
    2.258795574534202,
    2.3193465895257037,
    2.326647363773924,
    1.278336734622079,
    1.2613073596701336,
    1.262517914084326,
    1.263097817071652,
]
 #   1.262573331069125,
#]

x = np.arange(len(tuples))

fig, ax1 = plt.subplots(figsize=(8,5))

# sem_map: solid bar
ax1.bar(
    x,
    sem_map,
    width=0.6,
    color="#4C72B0",
    label="SemOp_1"
)

# sem_filter: transparent + hatch
ax1.bar(
    x,
    sem_filter,
    width=0.6,
    color="none",
    edgecolor="#D41717",
    hatch="///",
    linewidth=2,
    label="SemOp_2"
)

ax1.set_xlabel("# Tuples")
ax1.set_ylabel("Runtime (s)")
ax1.set_xticks(x)
ax1.set_xticklabels(tuples)

# throughput line
ax2 = ax1.twinx()
ax2.plot(
    x,
    throughput,
    color="#55A868",
    marker="o",
    linewidth=2,
    label="Throughput"
)

ax2.set_ylabel("Throughput (tuples/sec)")
ax2.set_ylim(bottom=0)

# move runtime legend slightly lower
ax1.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("motivation.pdf", dpi=300)
plt.show()