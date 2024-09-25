import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with np.load("results/results_all_methods_final_version_L16_N6000/gpu_par_filter_times.npz") as loaded:
    gpu_par_filter_mean_times = loaded["gpu_par_filter_mean_times"]
    gpu_par_filter_median_times = loaded["gpu_par_filter_median_times"]

with np.load("results/results_all_methods_final_version_L16_N6000/gpu_par_smooth_times.npz") as loaded:
    gpu_par_smooth_mean_times = loaded["gpu_par_smooth_mean_times"]
    gpu_par_smooth_median_times = loaded["gpu_par_smooth_median_times"]

with np.load("results/results_all_methods_final_version_L16_N6000/gpu_seq_filter_times.npz") as loaded:
    gpu_seq_filter_mean_times = loaded["gpu_seq_filter_mean_times"]
    gpu_seq_filter_median_times = loaded["gpu_seq_filter_median_times"]

with np.load("results/results_all_methods_final_version_L16_N6000/gpu_seq_smooth_times.npz") as loaded:
    gpu_seq_smooth_mean_times = loaded["gpu_seq_smooth_mean_times"]
    gpu_seq_smooth_median_times = loaded["gpu_seq_smooth_median_times"]

n_length_space = 4
l = 2**n_length_space
N = 6000
T = N
lengths_space = np.logspace(n_length_space, int(np.log2(T)), num=10, base=2, dtype=int)


plt.loglog(lengths_space, gpu_par_filter_mean_times, label="par filter", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_par_smooth_mean_times, label="par smoother", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_seq_filter_mean_times, label="seq filter", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_seq_smooth_mean_times, label="seq smoother", linestyle="-.", linewidth=3)
plt.legend()
plt.title("GPU Mean Run Times")
plt.grid(True, which="both")
plt.show()


# data = np.stack([lengths_space,
#                  gpu_par_filter_mean_times,
#                  gpu_par_filter_median_times,
#                  gpu_par_smooth_mean_times,
#                  gpu_par_smooth_median_times,
#                  gpu_seq_filter_mean_times,
#                  gpu_seq_filter_median_times,
#                  gpu_seq_smooth_mean_times,
#                  gpu_seq_smooth_median_times], axis=1)
# columns = ["lengths_space",
#            "gpu_par_filter_mean_times",
#            "gpu_par_filter_median_times",
#            "gpu_par_smooth_mean_times",
#            "gpu_par_smooth_median_times",
#            "gpu_seq_filter_mean_times",
#            "gpu_seq_filter_median_times",
#            "gpu_seq_smooth_mean_times",
#            "gpu_seq_smooth_median_times"]
#
# df = pd.DataFrame(data, columns=columns)
# df.to_csv("results/results_all_methods_final_version_L16_N6000/gpu-final-l16N6000.csv")

