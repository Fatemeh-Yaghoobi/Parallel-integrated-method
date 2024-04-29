import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with np.load("results/gpu_par_filter_times.npz") as loaded:
    gpu_par_filter_mean_times = loaded["gpu_par_filter_mean_times"]
    gpu_par_filter_median_times = loaded["gpu_par_filter_median_times"]

with np.load("results/gpu_par_smooth_times.npz") as loaded:
    gpu_par_smooth_mean_times = loaded["gpu_par_smooth_mean_times"]
    gpu_par_smooth_median_times = loaded["gpu_par_smooth_median_times"]

with np.load("results/gpu_seq_filter_times.npz") as loaded:
    gpu_seq_filter_mean_times = loaded["gpu_seq_filter_mean_times"]
    gpu_seq_filter_median_times = loaded["gpu_seq_filter_median_times"]

with np.load("results/gpu_seq_smooth_times.npz") as loaded:
    gpu_seq_smooth_mean_times = loaded["gpu_seq_smooth_mean_times"]
    gpu_seq_smooth_median_times = loaded["gpu_seq_smooth_median_times"]

T = 5000
lengths_space = np.logspace(2, int(np.log2(T)), num=10, base=2, dtype=int)

data = np.stack([lengths_space,
                 gpu_par_filter_mean_times,
                 gpu_par_filter_median_times,
                 gpu_par_smooth_mean_times,
                 gpu_par_smooth_median_times,
                 gpu_seq_filter_mean_times,
                 gpu_seq_filter_median_times,
                 gpu_seq_smooth_mean_times,
                 gpu_seq_smooth_median_times], axis=1)
columns = ["lengths_space",
           "gpu_par_filter_mean_times",
           "gpu_par_filter_median_times",
           "gpu_par_smooth_mean_times",
           "gpu_par_smooth_median_times",
           "gpu_seq_filter_mean_times",
           "gpu_seq_filter_median_times",
           "gpu_seq_smooth_mean_times",
           "gpu_seq_smooth_median_times"]

df = pd.DataFrame(data, columns=columns)
df.to_csv("results/gpu.csv")


# plt.loglog(lengths_space, gpu_par_filter_mean_times, label="par filter", linestyle="-.", linewidth=3)
# plt.loglog(lengths_space, gpu_par_smooth_mean_times, label="par smoother", linestyle="-.", linewidth=3)
# plt.loglog(lengths_space, gpu_seq_filter_mean_times, label="seq filter", linestyle="-.", linewidth=3)
# plt.loglog(lengths_space, gpu_seq_smooth_mean_times, label="seq smoother", linestyle="-.", linewidth=3)
# plt.legend()
# plt.title("GPU Mean Run Times")
# plt.grid(True, which="both")
# plt.show()


