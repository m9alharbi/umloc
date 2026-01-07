import pandas as pd
import matplotlib.pyplot as plt
from os import path as osp
import numpy as np
from scipy.interpolate import interp1d


# # Minimal IEEE‑style defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.figsize": (4, 3.5),     # width matches IEEE single column
    "figure.dpi": 300,
    "axes.labelsize": 10,
    "lines.linewidth": 1.0,
    # "xtick.labelsize": 18,
    # "ytick.labelsize": 18,
    # "legend.fontsize": 16,
})

def load_and_plot(csv_path, output_directory, data):
    """
    IEEE‑style comparison plots of smartphone IMU vs. ZED IMU.

    Columns expected
    ----------------
    Smartphone IMU
        linear_acceleration_{x,y,z}
        angular_velocity_{x,y,z}

    ZED camera IMU
        imu_linear_acc_{x,y,z}
        imu_angular_vel_{x,y,z}
    """
    df = pd.read_csv(csv_path)

    # Time axis (s) relative to start
    t = (df["time_stamp_spy"] - df["time_stamp_spy"].iloc[0]) / 1_000.0

    
    # ---------- Linear acceleration ----------
    fig1, (ax1, ax2) = plt.subplots(2, 1)
    for axis in ("x", "y", "z"):
        ax1.plot(t, df[f"linear_acceleration_{axis}"],
                 label=rf"Phone $a_{axis}$")
        ax2.plot(t, df[f"imu_linear_acc_{axis}"],
                 linestyle="--",
                 label=rf"ZED $a_{axis}$")


    ax1.grid(True, linewidth=0.25)
    ax1.legend(ncol=2, frameon=False)
    ax2.grid(True, linewidth=0.25)
    ax2.legend(ncol=2, frameon=False)



    fig1.tight_layout()
    plt.savefig(osp.join(output_directory, '{}_{}.eps'.format(data, 'acceleration')))


    # ---------- Angular velocity ----------
    fig2, (ax3, ax4) = plt.subplots(2, 1)
    for axis in ("x", "y", "z"):
        ax3.plot(t, df[f"angular_velocity_{axis}"],
                 label=rf"Phone $\omega_{axis}$")
        ax4.plot(t, df[f"imu_angular_vel_{axis}"],
                 linestyle="--",
                 label=rf"ZED $\omega_{axis}$")


    ax3.grid(True, linewidth=0.25)
    ax3.legend(ncol=2, frameon=False)
    ax4.grid(True, linewidth=0.25)
    ax4.legend(ncol=2, frameon=False)


    fig2.tight_layout()
    plt.savefig(osp.join(output_directory, '{}_{}.eps'.format(data, 'angular_velocity')))

output_directory = "/ibex/user/alham0a/localization_project/results/paper_plots/"

file_name = 'b5l5_7'
position_data = pd.read_csv('/ibex/user/alham0a/localization_project/data/datasets/KAUST_dataset/new_data/250609_020017_ros_position_data.csv')
phone_data = pd.read_csv('/ibex/user/alham0a/localization_project/data/datasets/KAUST_dataset/new_data/imu_250609_020006.csv')
phone_data = phone_data[:-1]
phone_data['time_stamp_spy'] = phone_data['time_stamp_spy'].astype('int64')
phone_cols = phone_data.columns
position_cols = position_data.columns
position_data = position_data.sort_values('time_stamp_ros')
phone_data = phone_data.sort_values('time_stamp_spy')
position_data = position_data.interpolate(method='linear')
phone_data = phone_data.interpolate(method='linear')
phone_data = phone_data.dropna()

fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True)#, height_ratios=[50, 50])
sensors_timestamp = phone_data['time_stamp_spy'].values
position_timestamp = position_data['time_stamp_ros'].values
start = max(sensors_timestamp.min(), position_timestamp.min())
end = min(sensors_timestamp.max(), position_timestamp.max())
duration = (end - start)/1000
num_samples = int(duration * 60)
time_axis = np.linspace(start, end, num_samples)
phone_interp_func = interp1d(sensors_timestamp, phone_data.values.T, kind='linear', fill_value="extrapolate")
position_interp_func = interp1d(position_timestamp, position_data.values.T, kind='linear', fill_value="extrapolate")
interp_phone_data = phone_interp_func(time_axis)
interp_position_data = position_interp_func(time_axis)
interp_phone_data = pd.DataFrame(interp_phone_data.T, columns=phone_cols)
interp_position_data = pd.DataFrame(interp_position_data.T, columns=position_cols)
t = (time_axis- time_axis[0]) / 1000.0

events = {"start": 12, "stop": 240.0}
segments = [("", 12, 240.0)]#, ("walking", 18, 208), ("resting", 210, 238.0)]


def annotate(ax):
    # vertical event markers
    for name, ts in events.items():
        ax.axvline(ts, linestyle="--", linewidth=2.5)
        # ax.annotate(name, xy=(ts, ax.get_ylim()[1]),
                    # xytext=(2, -14), textcoords="offset points",
                    # ha="left", va="top", fontsize=9)
    # shaded segments
    # for label, a, b in segments:
    #     ax.axvspan(a, b, alpha=0.10)
    #     ax.text((a+b)/2, ax.get_ylim()[0], label, fontsize=8, ha="center", va="bottom")

# i=0
# widths = {"x": 2.0, "y": 1.6, "z": 1.3}
# styles = {"x": '-', "y": '-.', "z": '--'}
# for axis in ("x", "y", "z"):
#     ax1.plot(t[1000:2000], interp_phone_data[f"linear_acceleration_{axis}"][1000:2000],
#              label=rf"${axis}$", linewidth=widths[axis])#, linestyle=styles[axis])
#     ax2.plot(t[1000:2000], interp_phone_data[f"angular_velocity_{axis}"][1000:2000], linewidth=widths[axis])
#     # ax3.plot(t, interp_position_data[f"position_{axis}"], linewidth=widths[axis])
#     # ax3.set_xticks([])
#     i+=1
    
ax1.plot(t[5000:-7000], interp_phone_data[f"linear_acceleration_x"][5000:-7000], color='C0')
ax2.plot(t[5000:-7000], interp_phone_data[f"linear_acceleration_y"][5000:-7000], color='C1')
ax3.plot(t[5000:-7000], interp_phone_data[f"linear_acceleration_z"][5000:-7000], color='C2')

ax4.plot(t[5000:-7000], interp_phone_data[f"angular_velocity_x"][5000:-7000], color='C3')
ax5.plot(t[5000:-7000], interp_phone_data[f"angular_velocity_y"][5000:-7000], color='C4')
ax6.plot(t[5000:-7000], interp_phone_data[f"angular_velocity_z"][5000:-7000], color='C5')

# ax1.set_xticks([])
# ax1.set_yticks([])
# ax2.set_yticks([])
# ax2.set_yticks([])
# t0, y0 = t[5*60], interp_phone_data['linear_acceleration_x'][5*60]
# ax1.scatter([t0], [y0], s=18)               # optional marker
# ax1.annotate(
#     "peak",                                # label
#     xy=(t0, y0),                           # point to
#     xytext=(t0 + 1.0, y0 + 0.5),           # text location (data coords)
#     arrowprops=dict(arrowstyle="->", lw=1.0, shrinkA=2, shrinkB=2),
#     fontsize=9, ha="left", va="bottom"
# )
# ax1.axvline(18, linestyle="--", linewidth=0.9)
# ax1.axvline(208, linestyle="--", linewidth=0.9)
# ax1.axvspan(12, 208, alpha=0.20)
# ax1.axvspan(208, 240, alpha=0.20, color='gray')
# ax1.set_ylabel(r'Acceleration ($\mathregular{m/sec^2}$)')
# ax1.legend(ncol=3)
# ax2.axvline(18, linestyle="--", linewidth=0.9)
# ax2.axvline(208, linestyle="--", linewidth=0.9)
# ax2.axvspan(12, 208, alpha=0.20)
# ax2.axvspan(208, 240, alpha=0.20, color='gray')
# ax2.set_ylabel(r'Angular velocity ($\mathregular{rad/sec}$)')
# ax3.axvline(18, linestyle="--", linewidth=0.9)
# ax3.axvline(208, linestyle="--", linewidth=0.9)
# ax3.axvspan(12, 208, alpha=0.20)
# ax3.axvspan(208, 240, alpha=0.20, color='gray')
# ax3.set_ylabel(r'Position ($\mathregular{m}$)')
# ax3.set_yticks([0, 10, 20])
# annotate(ax1)
# annotate(ax2)
# annotate(ax3)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax5.axis('off')
ax6.axis('off')
fig1.tight_layout()
fig1.savefig(osp.join(output_directory, '{}_imu.png'.format(file_name)), transparent=True)

gt = np.load("./results/umgloc_kaust/traj/b5l5c2_3_gt.npy", allow_pickle=True)
q = np.load("./results/umgloc_kaust/traj/b5l5c2_3_quantiles_umgloc.npy", allow_pickle=True)

gt = np.array(gt)        # [T, 2]
q = np.array(q)          # [T, 2, 2]
T = gt.shape[0]
t = np.arange(T)


# For each coord, define lower/upper envelopes
x_q_low, x_q_high = q[:, 0, 0], q[:, 1, 0]

y_q_low, y_q_high = q[:, 0, 1], q[:, 1, 1]

# --- Plot X coordinate over time with band ---
fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[50, 50])
ax1.plot(t, gt[:, 0], color='red', label="GT x", linestyle="--")
ax1.plot(t, q[:, 0, 0], label="Quantile 1 x")
ax1.plot(t, q[:, 1, 0], label="Quantile 2 x")
ax1.fill_between(t, x_q_low, x_q_high, alpha=0.2, label="x quantile band")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.axis('off')

# --- Plot Y coordinate over time with band ---
ax2.plot(t, gt[:, 1], color='red', label="GT y", linestyle="--")
ax2.plot(t, q[:, 0, 1], label="Quantile 1 y")
ax2.plot(t, q[:, 1, 1], label="Quantile 2 y")
ax2.fill_between(t, y_q_low, y_q_high, alpha=0.2, label="y quantile band")
ax2.axis('off')

fig2.tight_layout()
fig2.savefig(osp.join(output_directory, '{}_bounds.png'.format(file_name)))



# # Execute on the uploaded dataset
# data = "b9l4_2classes_2"
# load_and_plot("/ibex/user/alham0a/localization_project/data/datasets/umgloc_dataset/b9l4_2classes_2.csv", output_directory, data)
