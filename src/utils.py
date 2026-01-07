from os import path as osp
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import ndimage as ndi
from typing import List
from sklearn.model_selection import KFold   # pip install scikit-learn
from pathlib import Path

def split_sequences_kfold(txt_path, out_dir, k=10, make_val=True, shuffle=True):
    """
    Split a list of sequence IDs into K folds and save train/test files for each fold.

    Parameters
    ----------
    txt_path : str
        Path to the master .txt file (one ID per line).
    k        : int, default 5
        Number of folds.
    shuffle  : bool, default True
        Whether to shuffle IDs before splitting (recommended).
    seed     : int, default 42
        RNG seed used when `shuffle=True`.
    out_dir  : str | Path, default "."
        Directory where the per-fold txt files will be written.

    Returns
    -------
    List[tuple[str, str]]
        A list with one `(train_file, test_file)` tuple per fold, useful for logging.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read and clean the master list
    with open(txt_path, "r", encoding="utf-8") as f:
        seqs = [ln.strip() for ln in f if ln.strip()]
    if make_val and k < 3:
        raise ValueError("Need k ≥ 3 when make_val=True (train/val/test).")
    if len(seqs) < k:
        raise ValueError(f"Need at least {k} sequences; found {len(seqs)}.")

    # 2) Pre-compute fold indices
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=42)
    folds = [idx for _, idx in kf.split(seqs)]        # list of numpy arrays

    # 3) Iterate over folds
    saved: List[tuple[str, str, str]] = []
    for fold_no in range(k):
        test_idx = folds[fold_no]                     # current fold ➜ test
        val_idx  = folds[(fold_no + 1) % k] if make_val else []  # next fold ➜ val
        train_idx = [i for j, f in enumerate(folds)
                     if j not in {fold_no, (fold_no + 1) % k if make_val else -1}
                     for i in f]

        train_seqs = [seqs[i] for i in train_idx]
        val_seqs   = [seqs[i] for i in val_idx]
        test_seqs  = [seqs[i] for i in test_idx]

        # 4) Write files
        train_file = out_dir / f"train_fold{fold_no+1}.txt"
        val_file   = out_dir / f"val_fold{fold_no+1}.txt"   if make_val else ""
        test_file  = out_dir / f"test_fold{fold_no+1}.txt"

        train_file.write_text("\n".join(train_seqs) + "\n", encoding="utf-8")
        if make_val:
            Path(val_file).write_text("\n".join(val_seqs) + "\n", encoding="utf-8")
        test_file.write_text("\n".join(test_seqs) + "\n", encoding="utf-8")

        print(f"Fold {fold_no+1:>2}: "
              f"{len(train_seqs):>2} train  "
              f"{len(val_seqs):>2} val  "
              f"{len(test_seqs):>2} test")

        saved.append((str(train_file),
                      str(val_file) if make_val else "",
                      str(test_file)))

    return saved

def perturb_imu(
    x: torch.Tensor,
    noise_std: float = 0.05,
    drop_p: float = 0.10,
    *,
    per_feature: bool = True,
    drop_value: float = 0.0,
) -> torch.Tensor:
    """
    Apply two quick robustness perturbations:
      1. Gaussian 'noise burst'
      2. Random packet drop (zeros)

    Args
    ----
    x           : IMU batch (B, T, C) – **unchanged in place**
    noise_std   : σ multiplier for noise; σ=0.05 → 5 % of feature std-dev
    drop_p      : probability that an entire sample is dropped
    per_feature : if True, scale noise by each channel's std; else by global std
    drop_value  : value to insert when a sample is 'dropped' (default 0)

    Returns
    -------
    x_perturbed : new tensor with same dtype/device/shape
    """
    # 1) --------  Gaussian noise burst  --------
    if per_feature:
        scale = x.std(dim=(0, 1), keepdim=True)      # (1, 1, C)
    else:
        scale = x.std().view(1, 1, 1)                # (1, 1, 1)

    noise = torch.randn_like(x) * (noise_std * scale)
    x_noisy = x + noise

    # 2) --------  Random packet drop  --------
    if drop_p > 0:
        mask = torch.rand_like(x_noisy[..., 0]) > drop_p    # (B, T)
        mask = mask.unsqueeze(-1)                           # (B, T, 1) – broadcast
        x_noisy = torch.where(mask, x_noisy, torch.full_like(x_noisy, drop_value))

    return x_noisy

def apply_opening(mask, kernel_size=2):
    struct = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), kernel_size)
    return ndi.binary_opening(mask, structure=struct)

def apply_closing(mask, kernel_size=2):
    struct = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), kernel_size)
    return ndi.binary_closing(mask, structure=struct)

def apply_opening_then_closing(mask, kernel_size=2):
    return apply_closing(apply_opening(mask, kernel_size), kernel_size)

def apply_gaussian_smoothing_then_threshold(mask, sigma=1.0, thresh=0.6):
    smoothed = ndi.gaussian_filter(mask.astype(float), sigma=sigma)
    return smoothed > thresh


def relative_to_abs(rel_traj, start_pos, dts, target_type='global_vel'):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    # pos = rel_traj.clone()
    if target_type == 'global_vel':
        pos = rel_traj * dts
        pos[:, 0, :] = start_pos
        abs_traj = torch.cumsum(pos, dim=1)
    else:
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos
    return abs_traj

def write_config(args):
    if args.output_directory:
        with open(osp.join(args.output_directory, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "pytorch_global_position"
            json.dump(values, f, sort_keys=True)

def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep
    return result[:-1]

def plot_2d_uncertainty_ellipses(x, y, covariances, n_std=1.64, ax=None,
                                 facecolor='none', edgecolor='red', alpha=0.5, label='Uncertainty Ellipse'):
    """
    Plot 2D uncertainty ellipses for each (x, y) point with given covariances.

    Args:
        x (np.ndarray): Array of x coordinates (length T).
        y (np.ndarray): Array of y coordinates (length T).
        covariances (list or np.ndarray): List/array of 2x2 covariance matrices (length T).
        n_std (float): Number of standard deviations (e.g., 1.64 ≈ 90% confidence, 2 ≈ 95% confidence).
        ax (matplotlib axis, optional): Axis to plot on. If None, create a new figure.
        facecolor (str): Fill color inside ellipses.
        edgecolor (str): Edge color of ellipses.
        alpha (float): Transparency of ellipses.
        label (str): Label for the first ellipse (others will be silent).

    Returns:
        matplotlib axis with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 10))
    
    first = True  # To only label the first ellipse
    for xi, yi, cov in zip(x, y, covariances):
        # Ensure the covariance matrix is symmetric and positive semi-definite
        cov = np.array(cov)
        if cov.shape != (2, 2):
            raise ValueError("Each covariance matrix must be 2x2.")

        # Eigen decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        # Compute ellipse properties
        angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigenvals)

        ellipse = Ellipse(
            xy=(xi, yi),
            width=width,
            height=height,
            angle=angle,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            label=label if first else None  # Only label the first ellipse
        )
        ax.add_patch(ellipse)
        first = False

    ax.set_aspect('equal')
    return ax

def compute_covariances_from_bounds(x_lower, x_upper, y_lower, y_upper, confidence_level=0.90):
    """
    Compute 2x2 covariance matrices at each point based on lower and upper bounds.

    Args:
        x_lower, x_upper: Arrays of x-axis lower and upper bounds.
        y_lower, y_upper: Arrays of y-axis lower and upper bounds.
        confidence_level: Confidence level (e.g., 0.90 for 90%).

    Returns:
        List of 2x2 covariance matrices (one per point).
    """
    assert len(x_lower) == len(x_upper) == len(y_lower) == len(y_upper), "Array lengths must match."

    # Z-score corresponding to the confidence interval
    from scipy.stats import norm
    z_score = norm.ppf(0.5 + confidence_level / 2)

    covariances = []
    for xl, xu, yl, yu in zip(x_lower, x_upper, y_lower, y_upper):
        # Estimate standard deviations
        sigma_x = (xu - xl) / (2 * z_score)
        sigma_y = (yu - yl) / (2 * z_score)

        # Build covariance matrix (assuming no correlation between x and y)
        cov = np.array([
            [sigma_x**2, 0],
            [0, sigma_y**2]
        ])
        covariances.append(cov)

    return covariances
    
def plot_result(preds, gt, x_lower, y_lower, x_upper, y_upper, error, cumulative_prob, pos_cum_error, data, args):
    # Assuming the first column is 'x' and the second column is 'y' for both preds and gt
    time = np.linspace(0, len(preds)//30, len(preds))
    # breakpoint()
    # 2D plot of x and y axis
    plt.figure(figsize=(18, 10))
    plt.plot(preds[:, 0], preds[:, 1], label='Predictions', linewidth=3)
    plt.plot(gt[:, 0], gt[:, 1], label='Ground Truth', linewidth=3)
    plt.xticks(fontweight='bold', fontsize=22)
    plt.yticks(fontweight='bold', fontsize=22)
    plt.tick_params(axis='both', which='major', width=1)
    #plt.xlim((-20, 20))
    #plt.ylim([-20, 20])

    plt.grid(True)
    plt.legend(prop={'weight':'bold', 'size': 22})

    if args.show_plot:
        plt.show()

    if args.output_directory is not None and osp.isdir(args.output_directory):
        plt.savefig(osp.join(args.output_directory, '{}_{}.eps'.format(data, '2D_lstm')))
    plt.close('all')

    # X-axis plot with respect to time
    plt.figure(figsize=(18, 10))
    plt.plot(time, preds[:, 0], label='Predictions', linewidth=5)
    plt.plot(time, gt[:, 0], label='Ground Truth', linewidth=5)
    plt.xticks(fontweight='bold', fontsize=22)
    plt.yticks(fontweight='bold', fontsize=22)
    plt.tick_params(axis='both', which='major', width=1)
    plt.grid(True)
    plt.legend(prop={'weight':'bold', 'size': 22})
    if args.show_plot:
        plt.show()

    if args.output_directory is not None and osp.isdir(args.output_directory):
        plt.savefig(osp.join(args.output_directory, '{}_{}.eps'.format(data, 'x_lstm')))
    plt.close('all')
    # Y-axis plot with respect to time
    plt.figure(figsize=(18, 10))
    plt.plot(time, preds[:, 1], label='Predictions', linewidth=5)
    plt.plot(time, gt[:, 1], label='Ground Truth', linewidth=5)
    plt.xticks(fontweight='bold', fontsize=22)
    plt.yticks(fontweight='bold', fontsize=22)
    plt.tick_params(axis='both', which='major', width=1)
    plt.grid(True)
    plt.legend(prop={'weight':'bold', 'size': 22})
    
    if args.show_plot:
        plt.show()

    if args.output_directory is not None and osp.isdir(args.output_directory):
        plt.savefig(osp.join(args.output_directory, '{}_{}.eps'.format(data, 'y_lstm')))
    plt.close('all')
    plt.figure(figsize=(18, 10))
    plt.plot(error, cumulative_prob, linewidth=5)
    plt.xticks(fontweight='bold', fontsize=22)
    plt.yticks(fontweight='bold', fontsize=22)
    plt.tick_params(axis='both', which='major', width=1)
    plt.grid(True)
    
    if args.show_plot:
        plt.show()

    if args.output_directory is not None and osp.isdir(args.output_directory):
        plt.savefig(osp.join(args.output_directory, '{}_{}.eps'.format(data, 'CDF_error_lstm')))
    plt.close('all')
    plt.figure(figsize=(18, 10))
    plt.plot(time, pos_cum_error, linewidth=5)
    plt.xticks(fontweight='bold', fontsize=22)
    plt.yticks(fontweight='bold', fontsize=22)
    plt.tick_params(axis='both', which='major', width=1)
    plt.grid(True)

    if args.show_plot:
        plt.show()

    if args.output_directory is not None and osp.isdir(args.output_directory):
        plt.savefig(osp.join(args.output_directory, '{}_{}.eps'.format(data, 'error_lstm')))
    plt.close('all')


    # plt.figure(figsize=(18, 10))

    # # Plot the main predicted trajectory
    # plt.plot(preds[:, 0], preds[:, 1], color='blue', label='Predicted Trajectory')
    # plt.plot(x_lower, y_lower, 'r--', label='Predicted lower')
    # plt.plot(x_upper, y_upper, 'g--', label='Predicted upper')
    # # plt.show()
    # # breakpoint()
    # # # Combine the upper and lower bounds to form a polygon
    # # bound_x = np.concatenate([x_lower, x_upper[::-1]])
    # # bound_y = np.concatenate([y_lower, y_upper[::-1]])

    # # # Fill the region between lower and upper bounds
    # # plt.fill(bound_x, bound_y, color='gray', alpha=0.1, label='90% Prediction Interval')
    # plt.xticks(fontweight='bold', fontsize=22)
    # plt.yticks(fontweight='bold', fontsize=22)
    # plt.tick_params(axis='both', which='major', width=1)
    # plt.xlim((-20, 20))
    # plt.ylim([-20, 20])

    # plt.grid(True)
    # plt.legend(prop={'weight':'bold', 'size': 22})

    # if args.show_plot:
    #     plt.show()

    # if args.output_directory is not None and osp.isdir(args.output_directory):
    #     plt.savefig(osp.join(args.output_directory, '{}_{}.eps'.format(data, '2D_wbound')))
    # plt.close('all')
