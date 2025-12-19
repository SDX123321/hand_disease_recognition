# plotter.py
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metrics_dict, track_id, figsize=(12, 9)):
    if not metrics_dict or not any(metrics_dict.values()):
        print("No metrics data to plot.")
        return

    euclid = np.array(metrics_dict["euclid"])
    cos = np.array(metrics_dict["cos"])
    angle = np.array(metrics_dict["angle"])
    frames = np.arange(len(euclid))

    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(f'Hand Motion Metrics (Track ID: {track_id})', fontsize=14)

    data_list = [
        (euclid, "Euclidean Distance (px)", "tab:blue"),
        (cos, "Cosine Similarity", "tab:orange"),
        (angle, "Turning Angle (deg)", "tab:green")
    ]

    for ax, (data, title, color) in zip(axs, data_list):
        if len(data) == 0:
            ax.set_title(title)
            continue

        ax.plot(frames, data, color=color, linewidth=1.2, label=title)

        mean_val = np.mean(data)
        max_val = np.max(data)
        min_val = np.min(data)
        max_idx = np.argmax(data)
        min_idx = np.argmin(data)

        ax.scatter([max_idx], [max_val], color='red', zorder=5)
        ax.scatter([min_idx], [min_val], color='purple', zorder=5)
        ax.axhline(mean_val, color='gray', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}')

        ax.text(max_idx, max_val, f"Max: {max_val:.2f}", color='red', fontsize=9,
                verticalalignment='bottom', horizontalalignment='center')
        ax.text(min_idx, min_val, f"Min: {min_val:.2f}", color='purple', fontsize=9,
                verticalalignment='top', horizontalalignment='center')

        ax.set_ylabel(title)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

    axs[-1].set_xlabel("Frame Index")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()