#!/usr/bin/env python3
"""
Complete evaluation pipeline for Uncertainty-Guided Map-Aware Inertial
Localisation  (metrics → accuracy table → CDF → trajectory map figure).

Typical call
------------
python pipeline.py \
    --sequences 01 02 03 \
    --data_dir  ./traj_npy \
    --map_dir   ./maps \
    --out_dir   ./results \
    --delta_rte 100 \
    --make_traj_plots \
    --make_cdf_plot \
    --latex
"""
# -------------- imports -----------------------------------------------------
from pathlib import Path
import argparse, textwrap, json
import numpy as np, pandas as pd, scipy.stats as st
import matplotlib as mpl, matplotlib.pyplot as plt
import yaml
from PIL import Image
from scipy import stats
from scipy import ndimage as ndi
import pdb
from collections import defaultdict
from matplotlib.ticker import LinearLocator, ScalarFormatter, MultipleLocator, FuncFormatter, FixedLocator
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import pickle
from utils import *

FIGSIZE = (4, 3.5
          )        # 80 mm single-column square
mpl.rcParams.update({
    # "figure.figsize": (9, 5),     # width matches IEEE single column
    "savefig.bbox": "tight",
    "ps.fonttype": 42,

    # Fonts (choose one family and be consistent)
    "font.size": 10,                   # ~9–10 pt at final size
    "figure.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    # If you prefer sans-serif: set family to 'sans-serif' and list ['Arial','Helvetica','DejaVu Sans']

    # Axis & ticks
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    # "xtick.labelsize": 16,
    # "ytick.labelsize": 16,
    "legend.fontsize": 6,

    # Lines & markers
    "lines.linewidth": 2.0,           # ≥0.5 pt at print; 1.0 is safe
    "lines.markersize": 8,
})

# -------------- load / save -----------------------------------------------
def load_traj(path_pred, path_gt):
    P, G = np.load(path_pred), np.load(path_gt)
    n = len(G)
    # if len(P) >= n:
    #     P = P[:n]
    if len(P) != n:
        t1 = np.linspace(0, G.shape[0], P.shape[0])
        t2 = np.linspace(0, G.shape[0], G.shape[0])
        P = interp1d(t1, P, axis=0)(t2)
        # pad = np.repeat(P[-1][None, :], n - len(P), axis=0)
        # P = np.vstack([P, pad])

    if P.shape[1] > 2:
        # breakpoint()
        P = P[:, :2]
    # assert P.shape == G.shape and P.shape[1] == 2, \
    #     f"Shape mismatch {path_pred.name}"
    return P, G

def load_map(pgm, yml):
    img = np.array(Image.open(pgm))[::-1, :]     # flip for matplotlib
    with open(yml) as f:
        ydata = yaml.safe_load(f)
    res, org   = ydata["resolution"], ydata["origin"][:2]
    free_thresh = ydata["free_thresh"]
    if ydata['negate'] == 0:
        img = (255 - img) / 255 # ROS convention occ=1 white (>occ_thresh), free=0 (<free_thresh) black 
    else:
        img = img / 255
    h, w       = img.shape
    extent     = [org[0], org[0]+w*res, org[1], org[1]+h*res]
    return img, extent, free_thresh
    
# -------------- metric helpers ---------------------------------------------
def _euclid(A, B):                # vectorised L2
    return np.linalg.norm(A - B, axis=1)


def fde(P, G):
    position_drift = np.linalg.norm(G[-1 :] - P[-1, :])
    delta_position = G[1:, :] - G[:-1, :]
    delta_length = np.linalg.norm(delta_position, axis=-1)
    moving_len = np.sum(delta_length)
    return position_drift #/ moving_len

def ate(P, G):
    # return np.sqrt(np.mean((np.linalg.norm(P - G, axis=1))**2))
    return np.sqrt(np.mean((P - G) ** 2))

def all_ate(P, G):
    return np.linalg.norm(P - G, axis=1)
    
def rte(P, G, delta=100, fs=60):
    max_delta = P.shape[0]
    delta = delta * fs
    deltas = np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = P[deltas[i]:] + G[:-deltas[i]] - P[:-deltas[i]] - G[deltas[i]:]
        # rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))
        rtes[i] = np.sqrt(np.mean(err ** 2))
    return np.mean(rtes)

def dte(est, gt, delta=1):
    """
    Almost the same as t_rte in which the length of a window is one minute, while the length of a window in d_rte is one meter(default).

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """

    gt_delta_len = np.linalg.norm(gt[1:] - gt[:-1], axis=1)
    end_index = np.zeros((est.shape[0], 1), dtype=int)

    # calculate where the 1 meter endpoint is
    j = 0
    i = 0
    current_sum = 0.0
    while i < est.shape[0]:
        while j < gt_delta_len.shape[0]:
            current_sum = current_sum + gt_delta_len[j]
            if current_sum >= 1.0:
                break
            j = j + 1
        if j == gt_delta_len.shape[0]:
            # done
            break
        else:
            # reach the endpoint x_{j+1} of x_i
            end_index[i] = j + 1
            current_sum = current_sum - gt_delta_len[j] # make sure current_sum < 1.0 now
            current_sum = current_sum - gt_delta_len[i]
            i = i + 1

    d_rtes = np.zeros(len(end_index))
    for i in range(len(end_index)):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[end_index[i]] + gt[i] - est[i] - gt[end_index[i]]
        # rtes[i] = np.sqrt(np.mean(err ** 2))
        d_rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(d_rtes)
    
def all_rte(P, G, delta=100, fs=60):
    max_delta = P.shape[0]

    delta = delta * fs
    deltas = np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    rtes = []#np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = P[deltas[i]:] + G[:-deltas[i]] - P[:-deltas[i]] - G[deltas[i]:]
        rtes.append(np.linalg.norm(err, axis=1))
    return rtes
    

def max_drift(drift_store):
    return (drift_store[1].max() / drift_store[0][-1]) * 100          # scalar


def drift_curve(gt, pred):
    """
    Returns (cum_dist, errors) where
        cum_dist : (N,) cumulative distance along GT
        errors   : (N,) L2 error at every step
    """
    # breakpoint()
    # dx = np.diff(trajectory_points[:, 0])
    # dy = np.diff(trajectory_points[:, 1])

    # # Calculate the step sizes (distances between consecutive points)
    # step_sizes = np.sqrt(dx**2 + dy**2)

    # # Sum the step sizes to get the total distance
    # total_distance = np.sum(step_sizes)

    step_d = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(step_d)))   # metres
    errors   = _euclid(pred, gt)
    
    return cum_dist, errors

# ---- QUANTILE CALIBRATION HELPERS -----------------------------------------
def calib_interval(lower, upper, gt, alpha=0.90):
    """
    lower, upper, gt : (T,2)
    returns (picp, aiw, iscore_vector) where
        picp  = empirical coverage
        aiw   = average interval width (L2)
        is    = interval score (vector, will be averaged later)
    """
    
    n = len(gt)
    if len(upper) >= n:
        upper = upper[:n]
        lower = lower[:n]
    else:
        m = len(upper)
        gt = gt[:m]
    inside = (gt[:,0] >= lower[:,0]) & (gt[:,0] <= upper[:,0]) & \
             (gt[:,1] >= lower[:,1]) & (gt[:,1] <= upper[:,1])
    picp = inside.mean()
    ece = abs(picp - alpha)          # args.alpha = 0.90 by default

    aiw = np.linalg.norm(upper - lower, axis=1).mean()

    below = (gt < lower).any(axis=1)
    above = (gt > upper).any(axis=1)
    miss  = below | above
    iscore = np.linalg.norm(upper-lower, axis=1)               # width term
    # add penalisation for misses (vectorised)
    iscore += (2/alpha) * np.linalg.norm(lower-gt, axis=1) * below
    iscore += (2/alpha) * np.linalg.norm(gt-upper, axis=1) * above
    return ece, aiw

def get_map_metrics(pred, pgm_path, yaml_path, out_dir, seq):
    """
    pred : (T,2) predicted trajectory in metres
    returns (free_ratio, mean_dist)
        free_ratio  = % of points falling in free cells
        mean_dist   = mean Euclidean distance [m] to nearest obstacle
    """
    img, extent, free_thresh = load_map(pgm_path, yaml_path)
    # binary_map = img < free_thresh
    #img = img/255
    # img_norm = (255-img)/255
    # free_mask = img_norm
    # obs_mask = img_norm
    # free_mask[img_norm<0.196] = 0
    # obs_mask[img_norm>0.65] = 1
    # free_mask = 1-free_mask
    # obs_mask = 1-obs_mask
    # breakpoint()
    res = 0.05 #(extent[1]-extent[0]) / img.shape[1]      # metres / pixel

    # binary mask: 1 = free, 0 = obstacle/unknown
    free_mask = (img < free_thresh).astype(np.uint8)

    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(1-img, cmap="gray", extent=extent, origin="lower")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out = out_dir / f"{seq}_binary_map.png"
    fig.savefig(out, format='png', bbox_inches='tight', dpi=300)

    # distance (pixels) from every free cell to nearest obstacle
    free_mask = apply_gaussian_smoothing_then_threshold(free_mask, sigma=1.0, thresh=0.2) #0.6
    free_mask = apply_opening_then_closing(free_mask, 2)
    dist_pix = ndi.distance_transform_edt(free_mask)
    dist_m   = dist_pix * res                       # convert to metres
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(dist_m, cmap="gray", extent=extent, origin="lower")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out = out_dir / f"{seq}_map.png"
    fig.savefig(out, format='png', bbox_inches='tight', dpi=300)

    # helper to map world→pixel indices
    def world_to_pix(xy):
        px = ((xy[:,0]-extent[0]) / res).astype(int)
        py = ((xy[:,1]-extent[2]) / res).astype(int)
        return px, py

    px, py = world_to_pix(pred)
    valid  = (px>=0)&(px<img.shape[1])&(py>=0)&(py<img.shape[0])
    px, py = px[valid], py[valid]

    inside_free = free_mask[py, px] == 1
    free_ratio  = inside_free.mean()

    # clip indices outside map to 0 distance
    #px = np.clip(px, 0, img.shape[1]-1)
    #py = np.clip(py, 0, img.shape[0]-1)
    # mean_dist = dist_m[py, px].mean()

    return free_ratio

# ---------- after summarise(df) ------------------------------------------------
def sig_tests(df, metrics, m1, m2):
    """
    Returns a DataFrame of p-values (paired t-test) comparing m1 with m2 & m3
    over the per-sequence metric vectors already in df.
    """
    rows = []
    for met in metrics:
        v1 = df.loc[df.Model==m1, met].values
        for other in m2:
            v2 = df.loc[df.Model==other, met].values
            if np.isnan(v1).all() or np.isnan(v2).all():
                p = np.nan
            else:
                p = stats.ttest_rel(v1, v2).pvalue         # paired t
            rows.append({'Metric':met, 'Compare':f'{m1} vs {other}', 'p':p})
    return pd.DataFrame(rows).set_index(['Metric','Compare'])
    
# -------------- tabular accuracy summary -----------------------------------

def summarise_small(df):
    """
    Return a tidy table with one entry per metric of the form
        mean ± std   (both to 3 decimals)
    keeping one row per model.
    """
    numeric = [c for c in df.columns if c not in ('Seq', 'Model')]

    # group and compute mean + std at once → multi-index columns
    grp = df.groupby('Model')[numeric].agg(['mean', 'std'])

    # build a new DataFrame with single-level columns
    compact = pd.DataFrame(index=grp.index)
    for metric in numeric:
        mu  = grp[(metric, 'mean')].map('{:.3f}'.format)
        sig = grp[(metric, 'std')].map('{:.3f}'.format)
        compact[metric] = mu + ' ± ' + sig

    return compact
    
# -------------- one-sequence trajectory figure -----------------------------
def make_traj_plot(seq, paths, map_dir, out_dir, tags, styles):
    """IEEE-style trajectory + map overlay (EPS)."""
    gt = np.load(paths['gt'])
    preds = {k: np.load(p) for k, p in paths.items() if k != 'gt'}
    # apply to every sample
    # s, R, t = procrustes_similarity(preds, G)
    # S = (s * S @ R.T) + t            # (N,T,2)
    
    # mean_smoothed = savgol_filter(samples_aligned.mean(0), window_length=7,
    #                               polyorder=3, axis=0)
    
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # optional map
    if map_dir is not None:
        pgm = map_dir / f"{seq}.pgm"
        yml = map_dir / f"{seq}.yaml"
        if pgm.exists() and yml.exists():
            img, extent, free_thresh = load_map(pgm, yml)
            ax.imshow(1-img, cmap="gray", extent=extent,
                      origin="lower", alpha=0.7)

    handles = []
    labels = []
    for tag in tags:
        h = ax.plot(preds[tag][:,0], preds[tag][:,1],
                label=tag, color=styles[tag])
        handles.append(h[0])
        labels.append(tag)
    h = ax.plot(gt[:,0], gt[:,1], '-.', color='C3', label='Ground Truth', zorder=0)
    handles.append(h[0])
    labels.append('Ground Truth')
    h = ax.scatter(gt[0,0],  gt[0,1],
           marker='o', c='C2', edgecolors='k',
           zorder=6,    label='Start')
    
    handles.append(h)
    labels.append('Start')
    # 2) END – red cross (or triangle)
    h = ax.scatter(gt[-1,0], gt[-1,1],
           marker='X', c='C3',  edgecolors='k',
           zorder=6,   label='End')

    handles.append(h)
    labels.append('End')


  
    plt.yticks(fontsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
    ax.set_ylim(np.round(extent[-2:]))


    plt.xticks(fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.set_xlim(np.round(extent[:-2]))

    if seq == 'b9l4_class_corr_6_rev':
        ax.set_ylim([-12, 12])

    # ax.set_xlim(x_lim[0], x_lim[-1])

    plt.tight_layout()
    out = out_dir / f"{seq}_traj.pdf"
    fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
    
    fig_leg = plt.figure(figsize=(1.5, 0.8))   # width x height in inches
    fig_leg.legend(handles, labels,
                   ncol=len(labels),           # all items on one row
                   loc='center',
                   frameon=False,
                   handlelength=2.5,
                   columnspacing=1.2)
    fig_leg.savefig(out_dir / f"{seq}_traj_legend.pdf",
                    bbox_inches='tight',
                    transparent=True)
    plt.close(fig_leg)
    plt.close(fig)            # tidy up before next loop

def aggregate_curves(curves):
    """Return stacked drift matrix resampled on a common grid [0, max_d]."""
    max_d = max(c[-1] for c, d in curves)          # longest distance
    common_x = np.linspace(0, max_d, 1000)          # 501-point grid

    resampled = []
    for dist, drift in curves:                     # per-sequence
        drift_interp = np.interp(common_x, dist, drift)
        resampled.append(drift_interp)
    # current_offset = 0
    # resampled = []
    
    # for dist, drift in curves:
    #     # Shift the individual distance array by the current offset
    #     shifted_dist = dist + current_offset
        
    #     # Interpolate this specific segment onto the global total grid
    #     # We use left/right fill values to ensure we don't overwrite other sections
    #     drift_interp = np.interp(common_x, shifted_dist, drift, left=np.nan, right=np.nan)
    #     resampled.append(drift_interp)
        
    #     # Increase offset for the next trajectory
    #     current_offset += dist[-1]
        
    return common_x, np.vstack(resampled)          # (Nseq, 501)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def procrustes_similarity(src, tgt):
    """
    src, tgt : (T,2) arrays (mean prediction, ground truth)
    Returns scale s, 2×2 rotation R, translation t.
    """
    # centre
    src_c = src - src.mean(0)
    tgt_c = tgt - tgt.mean(0)

    # scale (optional: force s=1 to keep scale)
    s = (np.linalg.norm(tgt_c) / np.linalg.norm(src_c))

    # rotation via SVD
    H   = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    R   = U @ Vt                        # 2×2
    if np.linalg.det(R) < 0:           # reflection fix
        Vt[-1] *= -1
        R = U @ Vt

    t = tgt.mean(0) - s*R @ src.mean(0)
    return s, R, t
    
def make_uncertainty_plots(seq, sample_path, gt_path,
                           map_dir, out_dir, sigma=2.0):
    """
    Two-panel figure:
      (L) KDE heat-map (viridis, clipped) + mean/GT trajectories.
      (R) Uncertainty tube (arc-length vs ±sigma radius).
    Legend is exported separately as _legend.pdf.
    """
    # ---------- load data ---------------------------------------------------
    S = np.load(sample_path)              # (N, T, 2)
    G = np.load(gt_path)                  # (T, 2)
    N, T, _ = S.shape
    mean_traj = S.mean(axis=0)
    # apply to every sample
    # s, R, t = procrustes_similarity(mean_traj, G)
    # S = (s * S @ R.T) + t            # (N,T,2)
    
    # mean_smoothed = savgol_filter(samples_aligned.mean(0), window_length=7,
                                  # polyorder=3, axis=0)
    # ---------- canvas ------------------------------------------------------
    # fig, (axL, axR) = plt.subplots(1, 2, figsize=FIGSIZE)
    #                                gridspec_kw={'wspace': .28})
    fig, axL = plt.subplots(figsize=FIGSIZE)
    # ===== (L) KDE heat-map ================================================
    # (optional) occupancy map underlay
    if map_dir is not None:
        pgm = map_dir / f"{seq}.pgm"
        yml = map_dir / f"{seq}.yaml"
        if pgm.exists() and yml.exists():
            img, extent, _ = load_map(pgm, yml)
            axL.imshow(1-img, cmap='gray', extent=extent,
                       origin='lower', alpha=0.7)

    
    # KDE
    xy   = S.reshape(-1, 2).T
    kde  = stats.gaussian_kde(xy, bw_method='scott')
    xmin, xmax, ymin, ymax = extent
    # xmin, ymin = xy.min(axis=1); xmax, ymax = xy.max(axis=1)
    # Build a moderate grid for speed
    gx, gy = np.mgrid[xmin:xmax:256j, ymin:ymax:256j]
    prob   = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

    # Clip top 3 % to avoid one dark blob hiding detail
    vmax = np.percentile(prob, 97)
    # cmap = plt.get_cmap('hot')
    pcm  = axL.contourf(gx, gy, prob,
                        np.linspace(0, vmax, 25), cmap='viridis',
                        alpha=.65)

    # Mean + GT paths
    # h_mean, = axL.plot(mean_traj[:, 0], mean_traj[:, 1],
                       # color='yellow', lw=2.2, label='Mean')
    h_gt,   = axL.plot(G[:, 0],   G[:, 1], '-.',
                       color='C3', label='Ground Truth', zorder=0)
    h = axL.scatter(G[0,0],  G[0,1],
           marker='o', c='C2', edgecolors='k',
           zorder=6,    label='Start')
    
    # 2) END – red cross (or triangle)
    h = axL.scatter(G[-1,0], G[-1,1],
           marker='X', c='C3',  edgecolors='k',
           zorder=6,   label='End')

    
    plt.xticks(fontsize=14)
    axL.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axL.set_xlim(np.round(extent[:-2]))
    plt.yticks(fontsize=14)
    axL.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axL.set_ylim(np.round(extent[-2:]))


    plt.tight_layout()
    fig.savefig(out_dir / f'{seq}_uncert.pdf',
                format='pdf', bbox_inches='tight', dpi=300)
    with open(out_dir / f'{seq}_uncert.pickle', "wb") as f:
        pickle.dump(fig,f)
    plt.close()
    # errors = []
    # for i in S:
    #     errors.append(drift_curve(G, i)[-1])

    # error = np.vstack(errors)


def get_calib(data_list, tags, model_name, data_dir, styles, out_dir):
    plot_picp = {tag: [] for tag in tags}
    plot_aiw = {tag: [] for tag in tags}
    q_paths = {}
    for seq in data_list:
        G = {}
        Q = {}
        lower, upper = {}, {}
        paths = {'gt': data_dir/f'{seq}_gt.npy'}
        for tag in model_name.keys():
            q_paths.update({tag: data_dir/f'{seq}_quantiles_{model_name[tag]}.npy'})
            G[tag] = np.load(paths['gt'])
            Q[tag] = np.load(q_paths[tag])
            lower[tag], upper[tag] = Q[tag][:,0,:], Q[tag][:,1,:]

            if '68' in tag:
                alpha = 68
            elif '90' in tag:
                alpha = 90
            elif '95' in tag:
                alpha = 95
                
            picp, aiw = calib_interval(lower[tag], upper[tag], G[tags[0]], alpha)
            plot_picp[tag].append(picp), plot_aiw[tag].append(aiw)

    # Compute mean and std for each severity
    severities, mean_picp, std_picp = [], [], []
    stats_dict = defaultdict(list)
    for key, values in plot_picp.items():
        model, sev_str = key.split('_perturb_')
        PI = float(sev_str[:2])
        severity = float(sev_str[3:])
        stats_dict[model].append((PI, severity, np.mean(values), np.std(values)))

    stats_aiw = defaultdict(list)
    for key, values in plot_aiw.items():
        model, sev_str = key.split('_perturb_')
        PI = float(sev_str[:2])
        severity = float(sev_str[3:])
        stats_aiw[model].append((PI, severity, np.mean(values), np.std(values)))
        
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles = []
    labels = []
    width = 0.25
    # for model, data in stats_dict.items():
    # sort by severity
    # multiplier = 0
    model_rnin = 'RNIN'
    data_rnin = stats_dict[model_rnin]
    data_rnin.sort(key=lambda x: x[0])
    pi_rnin, sev_rnin, mean_rnin, std_rnin = zip(*data_rnin)
    x = np.arange(0, len(sev_rnin[:5]))
    h = ax.bar(x - width, mean_rnin[:5], width, label=model_rnin)
    
    handles.append(h[0])
    labels.append(model_rnin)
        
    model_umgloc = 'UMGLoc'
    data_umgloc = stats_dict[model_umgloc]
    data_umgloc.sort(key=lambda x: x[0])
    pi_umgloc, sev_umgloc, mean_umgloc, std_umgloc = zip(*data_umgloc)
    h = ax.bar(x, mean_umgloc[:5], width, label=model_umgloc)
    handles.append(h[0])
    labels.append(model_umgloc)

    model_umgloc_no_map = 'UMGLoc_no_map'
    data_umgloc_no_map = stats_dict[model_umgloc_no_map]
    data_umgloc_no_map.sort(key=lambda x: x[0])
    pi_umgloc_no_map, sev_umgloc_no_map, mean_umgloc_no_map, std_umgloc_no_map = zip(*data_umgloc_no_map)
    h = ax.bar(x + width, mean_umgloc_no_map[:5], width, label=model_umgloc_no_map)
    handles.append(h[0])
    labels.append(model_umgloc_no_map)
    # ax.bar(np.array(sev[6:12]) + offset + 0.2, mean[6:12], width)
    # ax.bar(np.array(sev[12:]) + offset + 0.4, mean[12:], width)
    # h = ax.plot(sev, mean, marker='o', linestyle='-', color=styles[model], label=model)
    # ax.fill_between(sev, np.array(mean) - np.array(std), np.array(mean) + np.array(std), alpha=0.15, color=styles[model])
    # handles.append(h[0])
    h = ax.axhline(y=68, color='r', linestyle='--', label=f'Nominal')
    handles.append(h)
    labels.append('Nominal')
    # Reference line
    # ax.axhline(0.9, linestyle='--', color='k', label='Nominal 0.9')
    

    ax.set_ylim(0, 100.0)
    ax.grid(True)
    ax.set_xticks(x, np.array(sev_umgloc_no_map[:5]))

    plt.tight_layout()
    out = out_dir / f"picp_68_plot.pdf"
    # plt.legend(loc='upper center', ncol=2)
    fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    fig_leg = plt.figure(figsize=(1.5, 0.8))   # width x height in inches
    fig_leg.legend(handles, labels,
                   ncol=len(labels),           # all items on one row
                   loc='center',
                   frameon=False,
                   handlelength=2.5,
                   columnspacing=1.2)
    fig_leg.savefig(out_dir / f"robustness_legend.pdf",
                    bbox_inches='tight',
                    transparent=True)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    model_rnin = 'RNIN'
    data_rnin = stats_dict[model_rnin]
    data_rnin.sort(key=lambda x: x[0])
    pi_rnin, sev_rnin, mean_rnin, std_rnin = zip(*data_rnin)
    x = np.arange(0, len(sev_rnin[6:11]))
    ax.bar(x - width, mean_rnin[6:11], width, label=model_rnin)
    
    model_umgloc = 'UMGLoc'
    data_umgloc = stats_dict[model_umgloc]
    data_umgloc.sort(key=lambda x: x[0])
    pi_umgloc, sev_umgloc, mean_umgloc, std_umgloc = zip(*data_umgloc)
    ax.bar(x, mean_umgloc[6:11], width, label=model_umgloc)
    
    model_umgloc_no_map = 'UMGLoc_no_map'
    data_umgloc_no_map = stats_dict[model_umgloc_no_map]
    data_umgloc_no_map.sort(key=lambda x: x[0])
    pi_umgloc_no_map, sev_umgloc_no_map, mean_umgloc_no_map, std_umgloc_no_map = zip(*data_umgloc_no_map)
    ax.bar(x + width, mean_umgloc_no_map[6:11], width, label=model_umgloc_no_map)
    # ax.bar(np.array(sev[6:12]) + offset + 0.2, mean[6:12], width)
    # ax.bar(np.array(sev[12:]) + offset + 0.4, mean[12:], width)
    # h = ax.plot(sev, mean, marker='o', linestyle='-', color=styles[model], label=model)
    # ax.fill_between(sev, np.array(mean) - np.array(std), np.array(mean) + np.array(std), alpha=0.15, color=styles[model])
    # handles.append(h[0])
    ax.axhline(y=90, color='r', linestyle='--', label=f'Nominal {90}%')
    
    # Reference line
    # ax.axhline(0.9, linestyle='--', color='k', label='Nominal 0.9')
    

    ax.set_ylim(0, 100.0)
    ax.grid(True)
    ax.set_xticks(x, np.array(sev_umgloc_no_map[6:11]))
    plt.tight_layout()
    out = out_dir / f"picp_90_plot.pdf"
    fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    model_rnin = 'RNIN'
    data_rnin = stats_dict[model_rnin]
    data_rnin.sort(key=lambda x: x[0])
    pi_rnin, sev_rnin, mean_rnin, std_rnin = zip(*data_rnin)
    x = np.arange(0, len(sev_rnin[12:-1]))
    ax.bar(x - width, mean_rnin[12:-1], width, label=model_rnin)
    
    model_umgloc = 'UMGLoc'
    data_umgloc = stats_dict[model_umgloc]
    data_umgloc.sort(key=lambda x: x[0])
    pi_umgloc, sev_umgloc, mean_umgloc, std_umgloc = zip(*data_umgloc)
    ax.bar(x, mean_umgloc[12:-1], width, label=model_umgloc)
    
    model_umgloc_no_map = 'UMGLoc_no_map'
    data_umgloc_no_map = stats_dict[model_umgloc_no_map]
    data_umgloc_no_map.sort(key=lambda x: x[0])
    pi_umgloc_no_map, sev_umgloc_no_map, mean_umgloc_no_map, std_umgloc_no_map = zip(*data_umgloc_no_map)
    ax.bar(x + width, mean_umgloc_no_map[12:-1], width, label=model_umgloc_no_map)
    # ax.bar(np.array(sev[6:12]) + offset + 0.2, mean[6:12], width)
    # ax.bar(np.array(sev[12:]) + offset + 0.4, mean[12:], width)
    # h = ax.plot(sev, mean, marker='o', linestyle='-', color=styles[model], label=model)
    # ax.fill_between(sev, np.array(mean) - np.array(std), np.array(mean) + np.array(std), alpha=0.15, color=styles[model])
    # handles.append(h[0])
    ax.axhline(y=95, color='r', linestyle='--', label=f'Nominal {95}%')

    # Reference line
    # ax.axhline(0.9, linestyle='--', color='k', label='Nominal 0.9')
    

    ax.set_ylim(0, 110.0)
    ax.grid(True)
    ax.set_xticks(x, np.array(sev_umgloc_no_map[12:-1]))

    plt.tight_layout()
    out = out_dir / f"picp_95_plot.pdf"
    fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    # fig_leg = plt.figure(figsize=(3.5, 1))   # width x height in inches
    # fig_leg.legend(handles, labels,
    #            ncol=len(labels),           # all items on one row
    #            loc='center',
    #            frameon=False,
    #            handlelength=2.5,
    #            columnspacing=1.2)
    # fig_leg.savefig(out_dir / f"{seq}_picp_legend.pdf",
    #             bbox_inches='tight',
    #             transparent=True)
    
    # plt.close(fig)
    # plt.close(fig_leg)


   
    # Prepare plot

    # for model, data in stats_dict.items():
    # sort by severity
    # multiplier = 0
    fig, ax = plt.subplots(figsize=FIGSIZE)
    model_rnin = 'RNIN'
    data_rnin = stats_aiw[model_rnin]
    data_rnin.sort(key=lambda x: x[0])
    pi_rnin, sev_rnin, mean_rnin, std_rnin = zip(*data_rnin)
    ax.bar(x - 0.25, mean_rnin[:5], width, label=model_rnin)

    
    model_umgloc = 'UMGLoc'
    data_umgloc = stats_aiw[model_umgloc]
    data_umgloc.sort(key=lambda x: x[0])
    pi_umgloc, sev_umgloc, mean_umgloc, std_umgloc = zip(*data_umgloc)
    ax.bar(x, mean_umgloc[:5], width, label=model_umgloc)

    
    model_umgloc_no_map = 'UMGLoc_no_map'
    data_umgloc_no_map = stats_aiw[model_umgloc_no_map]
    data_umgloc_no_map.sort(key=lambda x: x[0])
    pi_umgloc_no_map, sev_umgloc_no_map, mean_umgloc_no_map, std_umgloc_no_map = zip(*data_umgloc_no_map)
    ax.bar(x + 0.25, mean_umgloc_no_map[:5], width, label=model_umgloc_no_map)
    
    ax.set_xticks(x, np.array(sev_umgloc_no_map[:5]))

    plt.tight_layout()
    out = out_dir / f"aiw_68_plot.pdf"
    fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    model_rnin = 'RNIN'
    data_rnin = stats_aiw[model_rnin]
    data_rnin.sort(key=lambda x: x[0])
    pi_rnin, sev_rnin, mean_rnin, std_rnin = zip(*data_rnin)
    ax.bar(x-0.25, mean_rnin[6:11], width, label=model_rnin)

    
    model_umgloc = 'UMGLoc'
    data_umgloc = stats_aiw[model_umgloc]
    data_umgloc.sort(key=lambda x: x[0])
    pi_umgloc, sev_umgloc, mean_umgloc, std_umgloc = zip(*data_umgloc)
    ax.bar(x, mean_umgloc[6:11], width, label=model_umgloc)

    
    model_umgloc_no_map = 'UMGLoc_no_map'
    data_umgloc_no_map = stats_aiw[model_umgloc_no_map]
    data_umgloc_no_map.sort(key=lambda x: x[0])
    pi_umgloc_no_map, sev_umgloc_no_map, mean_umgloc_no_map, std_umgloc_no_map = zip(*data_umgloc_no_map)
    ax.bar(x + 0.25, mean_umgloc_no_map[6:11], width, label=model_umgloc_no_map)
    
    ax.set_xticks(x, np.array(sev_umgloc_no_map[6:11]))

    plt.tight_layout()
    out = out_dir / f"aiw_90_plot.pdf"
    fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    model_rnin = 'RNIN'
    data_rnin = stats_aiw[model_rnin]
    data_rnin.sort(key=lambda x: x[0])
    pi_rnin, sev_rnin, mean_rnin, std_rnin = zip(*data_rnin)
    ax.bar(x-0.25, mean_rnin[12:-1], width, label=model_rnin)

    
    model_umgloc = 'UMGLoc'
    data_umgloc = stats_aiw[model_umgloc]
    data_umgloc.sort(key=lambda x: x[0])
    pi_umgloc, sev_umgloc, mean_umgloc, std_umgloc = zip(*data_umgloc)
    ax.bar(x, mean_umgloc[12:-1], width, label=model_umgloc)

    
    model_umgloc_no_map = 'UMGLoc_no_map'
    data_umgloc_no_map = stats_aiw[model_umgloc_no_map]
    data_umgloc_no_map.sort(key=lambda x: x[0])
    pi_umgloc_no_map, sev_umgloc_no_map, mean_umgloc_no_map, std_umgloc_no_map = zip(*data_umgloc_no_map)
    ax.bar(x + 0.25, mean_umgloc_no_map[12:-1], width, label=model_umgloc_no_map)
    
    ax.set_xticks(x, np.array(sev_umgloc_no_map[12:-1]))
    plt.tight_layout()
    out = out_dir / f"aiw_95_plot.pdf"
    fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    
# -------------- main --------------------------------------------------------
def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    pa.add_argument('--sequences_list', required=True)
    pa.add_argument('--data_dir', required=True)
    pa.add_argument('--map_dir',  default=None)
    pa.add_argument('--out_dir',  default='./results')
    pa.add_argument('--delta_rte', type=int, default=60)
    pa.add_argument('--num_samples', type=int, default=10)
    pa.add_argument('--fs', type=int, default=60)

    pa.add_argument('--make_traj_plots', action='store_true')
    pa.add_argument('--make_p_test', action='store_true')
    pa.add_argument('--make_cdf_plot',  action='store_true')
    pa.add_argument('--make_calib_metrics', action='store_true')
    pa.add_argument('--make_drift_plot',  action='store_true')
    pa.add_argument('--make_map_metrics', action='store_true')
    pa.add_argument('--make_uncert_plots', action='store_true',
                help='Generate KDE+ellipse and uncertainty-tube plots')


    pa.add_argument('--alpha', type=float, default=0.90,
                help='Nominal coverage level of your interval (default 0.90)')

    pa.add_argument('--latex', action='store_true')
    args = pa.parse_args()
    data_dir = Path(args.data_dir)
    if args.map_dir is not None:
        map_dir  = Path(args.map_dir)
    else:
        map_dir = None
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # # ____________________________KAUST_________________________________
    # tags = ('RoNIN_TCN', 'RoNIN_LSTM_bi', 'RNIN', 'UMLoc_no_map', 'UMLoc')#, )
    # model_name = {'RoNIN_TCN': 'tcn_ronin', 'RoNIN_LSTM_bi': 'lstm_bi_ronin', 'RNIN':'rnin', 'UMLoc_no_map': 'umloc_nomap', 'UMLoc': 'umloc'}#, }

    # #_____________________________RONIN/RNN_____________________________
    # tags = ('UMGLoc', 'RoNIN_TCN', 'RoNIN_LSTM_bi', 'RNIN')
    # model_name = {'UMGLoc': 'umgloc_nomap', 'RoNIN_TCN': 'tcn_ronin', 'RoNIN_LSTM_bi': 'lstm_bi_ronin', 'RNIN':'rnin'}

    # ________________________Robustness__________________________
    # tags = ('UMGLoc_no_map_perturb_68_0.0', 'UMGLoc_no_map_perturb_68_0.1', 'UMGLoc_no_map_perturb_68_0.5', 
    #         'UMGLoc_no_map_perturb_68_1.0', 'UMGLoc_no_map_perturb_68_5.0', 'UMGLoc_no_map_perturb_68_10.0', 
    #         'UMGLoc_no_map_perturb_90_0.0', 'UMGLoc_no_map_perturb_90_0.1', 'UMGLoc_no_map_perturb_90_0.5', 
    #         'UMGLoc_no_map_perturb_90_1.0', 'UMGLoc_no_map_perturb_90_5.0', 'UMGLoc_no_map_perturb_90_10.0',
    #         'UMGLoc_no_map_perturb_95_0.0', 'UMGLoc_no_map_perturb_95_0.1', 'UMGLoc_no_map_perturb_95_0.5', 
    #         'UMGLoc_no_map_perturb_95_1.0', 'UMGLoc_no_map_perturb_95_5.0', 'UMGLoc_no_map_perturb_95_10.0',
    #         'UMGLoc_perturb_68_0.0', 'UMGLoc_perturb_68_0.1', 'UMGLoc_perturb_68_0.5', 
    #         'UMGLoc_perturb_68_1.0', 'UMGLoc_perturb_68_5.0', 'UMGLoc_perturb_68_10.0', 
    #         'UMGLoc_perturb_90_0.0', 'UMGLoc_perturb_90_0.1', 'UMGLoc_perturb_90_0.5', 
    #         'UMGLoc_perturb_90_1.0', 'UMGLoc_perturb_90_5.0', 'UMGLoc_perturb_90_10.0',
    #         'UMGLoc_perturb_95_0.0', 'UMGLoc_perturb_95_0.1', 'UMGLoc_perturb_95_0.5', 
    #         'UMGLoc_perturb_95_1.0', 'UMGLoc_perturb_95_5.0', 'UMGLoc_perturb_95_10.0',
    #         'RNIN_perturb_68_0.0', 'RNIN_perturb_68_0.1', 'RNIN_perturb_68_0.5', 
    #         'RNIN_perturb_68_1.0', 'RNIN_perturb_68_5.0', 'RNIN_perturb_68_10.0', 
    #         'RNIN_perturb_90_0.0', 'RNIN_perturb_90_0.1', 'RNIN_perturb_90_0.5', 
    #         'RNIN_perturb_90_1.0', 'RNIN_perturb_90_5.0', 'RNIN_perturb_90_10.0',
    #         'RNIN_perturb_95_0.0', 'RNIN_perturb_95_0.1', 'RNIN_perturb_95_0.5', 
    #         'RNIN_perturb_95_1.0', 'RNIN_perturb_95_5.0', 'RNIN_perturb_95_10.0')
    
    # model_name = {'UMGLoc_no_map_perturb_68_0.0': 'umgloc_nomap_perturb_68.0_0.0', 
    #               'UMGLoc_no_map_perturb_68_0.1': 'umgloc_nomap_perturb_68.0_0.1', 
    #               'UMGLoc_no_map_perturb_68_0.5': 'umgloc_nomap_perturb_68.0_0.5', 
    #               'UMGLoc_no_map_perturb_68_1.0': 'umgloc_nomap_perturb_68.0_1.0', 
    #               'UMGLoc_no_map_perturb_68_5.0': 'umgloc_nomap_perturb_68.0_5.0', 
    #               'UMGLoc_no_map_perturb_68_10.0': 'umgloc_nomap_perturb_68.0_10.0', 
    #               'UMGLoc_no_map_perturb_90_0.0': 'umgloc_nomap_perturb_90.0_0.0', 
    #               'UMGLoc_no_map_perturb_90_0.1': 'umgloc_nomap_perturb_90.0_0.1', 
    #               'UMGLoc_no_map_perturb_90_0.5': 'umgloc_nomap_perturb_90.0_0.5', 
    #               'UMGLoc_no_map_perturb_90_1.0': 'umgloc_nomap_perturb_90.0_1.0', 
    #               'UMGLoc_no_map_perturb_90_5.0': 'umgloc_nomap_perturb_90.0_5.0', 
    #               'UMGLoc_no_map_perturb_90_10.0': 'umgloc_nomap_perturb_90.0_10.0',
    #               'UMGLoc_no_map_perturb_95_0.0': 'umgloc_nomap_perturb_95.0_0.0', 
    #               'UMGLoc_no_map_perturb_95_0.1': 'umgloc_nomap_perturb_95.0_0.1', 
    #               'UMGLoc_no_map_perturb_95_0.5': 'umgloc_nomap_perturb_95.0_0.5', 
    #               'UMGLoc_no_map_perturb_95_1.0': 'umgloc_nomap_perturb_95.0_1.0', 
    #               'UMGLoc_no_map_perturb_95_5.0': 'umgloc_nomap_perturb_95.0_5.0', 
    #               'UMGLoc_no_map_perturb_95_10.0': 'umgloc_nomap_perturb_95.0_10.0',
    #               'UMGLoc_perturb_68_0.0': 'umgloc_perturb_68.0_0.0', 
    #               'UMGLoc_perturb_68_0.1': 'umgloc_perturb_68.0_0.1', 
    #               'UMGLoc_perturb_68_0.5': 'umgloc_perturb_68.0_0.5', 
    #               'UMGLoc_perturb_68_1.0': 'umgloc_perturb_68.0_1.0', 
    #               'UMGLoc_perturb_68_5.0': 'umgloc_perturb_68.0_5.0', 
    #               'UMGLoc_perturb_68_10.0': 'umgloc_perturb_68.0_10.0', 
    #               'UMGLoc_perturb_90_0.0': 'umgloc_perturb_90.0_0.0', 
    #               'UMGLoc_perturb_90_0.1': 'umgloc_perturb_90.0_0.1', 
    #               'UMGLoc_perturb_90_0.5': 'umgloc_perturb_90.0_0.5', 
    #               'UMGLoc_perturb_90_1.0': 'umgloc_perturb_90.0_1.0', 
    #               'UMGLoc_perturb_90_5.0': 'umgloc_perturb_90.0_5.0', 
    #               'UMGLoc_perturb_90_10.0': 'umgloc_perturb_90.0_10.0',
    #               'UMGLoc_perturb_95_0.0': 'umgloc_perturb_95.0_0.0', 
    #               'UMGLoc_perturb_95_0.1': 'umgloc_perturb_95.0_0.1', 
    #               'UMGLoc_perturb_95_0.5': 'umgloc_perturb_95.0_0.5', 
    #               'UMGLoc_perturb_95_1.0': 'umgloc_perturb_95.0_1.0', 
    #               'UMGLoc_perturb_95_5.0': 'umgloc_perturb_95.0_5.0', 
    #               'UMGLoc_perturb_95_10.0': 'umgloc_perturb_95.0_10.0',
    #               'RNIN_perturb_68_0.0': 'rnin_perturb_68_0.0', 
    #               'RNIN_perturb_68_0.1': 'rnin_perturb_68_0.1', 
    #               'RNIN_perturb_68_0.5': 'rnin_perturb_68_0.5', 
    #               'RNIN_perturb_68_1.0': 'rnin_perturb_68_1.0', 
    #               'RNIN_perturb_68_5.0': 'rnin_perturb_68_5.0', 
    #               'RNIN_perturb_68_10.0': 'rnin_perturb_68_10.0', 
    #               'RNIN_perturb_90_0.0': 'rnin_perturb_90_0.0', 
    #               'RNIN_perturb_90_0.1': 'rnin_perturb_90_0.1', 
    #               'RNIN_perturb_90_0.5': 'rnin_perturb_90_0.5', 
    #               'RNIN_perturb_90_1.0': 'rnin_perturb_90_1.0', 
    #               'RNIN_perturb_90_5.0': 'rnin_perturb_90_5.0', 
    #               'RNIN_perturb_90_10.0': 'rnin_perturb_90_10.0',
    #               'RNIN_perturb_95_0.0': 'rnin_perturb_95_0.0', 
    #               'RNIN_perturb_95_0.1': 'rnin_perturb_95_0.1', 
    #               'RNIN_perturb_95_0.5': 'rnin_perturb_95_0.5', 
    #               'RNIN_perturb_95_1.0': 'rnin_perturb_95_1.0', 
    #               'RNIN_perturb_95_5.0': 'rnin_perturb_95_5.0', 
    #               'RNIN_perturb_95_10.0': 'rnin_perturb_95_10.0'}
    # model_name = {'UMGLoc_perturb0.1': 'umgloc_perturb0.1', 'UMGLoc_perturb0.01': 'umgloc_perturb0.01',
    #               'UMGLoc_perturb0.001': 'umgloc_perturb0.001', 'UMGLoc_perturb0.5': 'umgloc_perturb0.5',
    #               'UMGLoc_perturb0.05': 'umgloc_perturb0.05', 'UMGLoc_perturb0.005': 'umgloc_perturb0.005',
    #               'RNIN_perturb0.1': 'rnin_perturb0.1', 'RNIN_perturb0.01': 'rnin_perturb0.01',
    #               'RNIN_perturb0.001': 'rnin_perturb0.001', 'RNIN_perturb0.5': 'rnin_perturb0.5',
    #               'RNIN_perturb0.05': 'rnin_perturb0.05', 'RNIN_perturb0.005': 'rnin_perturb0.005',
    #               'RoNIN_TCN_perturb0.1': 'tcn_ronin_perturb0.1', 'RoNIN_TCN_perturb0.01': 'tcn_ronin_perturb0.01',
    #               'RoNIN_TCN_perturb0.001': 'tcn_ronin_perturb0.001', 'RoNIN_TCN_perturb0.5': 'tcn_ronin_perturb0.5',
    #               'RoNIN_TCN_perturb0.05': 'tcn_ronin_perturb0.05', 'RoNIN_TCN_perturb0.005': 'tcn_ronin_perturb0.005',
    #               'RoNIN_LSTM_bi_perturb0.1': 'lstm_bi_ronin_perturb0.1', 'RoNIN_LSTM_bi_perturb0.01': 'tcn_ronin_perturb0.01',
    #               'RoNIN_LSTM_bi_perturb0.001': 'lstm_bi_ronin_perturb0.001', 'RoNIN_LSTM_bi_perturb0.5': 'lstm_bi_ronin_perturb0.5',
    #               'RoNIN_LSTM_bi_perturb0.05': 'lstm_bi_ronin_perturb0.05', 'RoNIN_LSTM_bi_perturb0.005': 'lstm_bi_ronin_perturb0.005'}

    # # ____________________________map-nomap___________________
    tags = ('UMLoc_no_map', 'UMLoc')
    model_name = {'UMLoc_no_map': 'umgloc_nomap', 'UMLoc': 'umgloc'}

    # tags = ('UMGLoc')
    # model_name = {'UMGLoc': 'umglocall'}
    # ---- iterate sequences --------------------------------------------------
    records = []
    cdf_drift = {tag: [] for tag in tags}
    cdf_ate = {tag: [] for tag in tags}
    cdf_rte = {tag: [] for tag in tags}
    cdf_all_ate = {tag: [] for tag in tags}
    cdf_all_rte = {tag: [] for tag in tags}
    plot_picp = {tag: [] for tag in tags}
    plot_aiw = {tag: [] for tag in tags}
    drift_store = {tag: [] for tag in tags}
    styles = {'UMGLoc': 'C0', 'UMGLoc_no_map': 'C1', 'RoNIN_TCN': 'C2', 'RoNIN_LSTM_bi': 'C4', 'RNIN': 'C5'}

    
    with open(args.sequences_list) as f:
        lines = [s.strip() for s in f.readlines() if len(s) > 0 and s[0] != '#']

    data_list = []
    for line in lines:
        data_name = line.split(',')[0]  # get the trajectory file name (without extension)
        data_list.append(data_name)

    if args.make_calib_metrics:
        get_calib(data_list, tags, model_name, data_dir, styles, out_dir)
    q_paths = {}
    for seq in data_list:
        paths = {'gt': data_dir/f'{seq}_gt.npy'}
        if args.make_uncert_plots:
            # multi-sample file must be (N,T,2); name it however you like
            sample_path = data_dir / f'{seq}_{model_name[tags]}.npy'
            # if not sample_path.exists():
            #     raise FileNotFoundError(sample_path)

            make_uncertainty_plots(seq,
                           sample_path   = sample_path,
                           gt_path       = paths['gt'],
                           map_dir       = map_dir,
                           out_dir       = out_dir,
                           sigma         = 2.0)
            continue
        for tag in model_name.keys():
            paths.update({tag: data_dir/f'{seq}_{model_name[tag]}.npy'})
            if args.make_calib_metrics:
                q_paths.update({tag: data_dir/f'{seq}_quantiles_{model_name[tag]}.npy'})


        # ---------- MAP CONSISTENCY -------------------------------------------
        if args.make_map_metrics:
            pgm = map_dir / f"{seq}.pgm"
            yml = map_dir / f"{seq}.yaml"
        if not all(p.exists() for p in paths.values()):
            raise FileNotFoundError(f'missing files for seq {seq}')

        # trajectory figure (optional)
        if args.make_traj_plots:
            make_traj_plot(seq, paths, map_dir, out_dir, tags, styles)
            
        
        # ---------- metrics & calibration ------------------------------------
        P, G = {}, {}
        for tag in model_name.keys():
            P[tag], G[tag] = load_traj(paths[tag], paths['gt'])

        # ---------- DRIFT ------------------------------------------------------
        if args.make_drift_plot:
            for tag in model_name.keys():
                drift_store[tag].append(drift_curve(G[tag], P[tag]))        # per-model curve
                # max drift-rate per 100m



        # --- quantile file only exists for our model (m1) --------------------
        if args.make_calib_metrics:
            Q = {}
            lower, upper = {}, {}
            for tag in model_name.keys():
                Q[tag] = np.load(q_paths[tag])
                if tag[:6] == 'UMGLoc':
                    lower[tag], upper[tag] = Q[tag][:,0,:], Q[tag][:,2,:]
                else:
                    lower[tag], upper[tag] = Q[tag][:,0,:], Q[tag][:,1,:]
        
                picp, aiw = calib_interval(lower[tag], upper[tag], G[tags[0]], alpha=args.alpha)
                plot_picp[tag].append(picp), plot_aiw[tag].append(aiw)

        if args.make_map_metrics:
            free, distance = {}, {}
            for tag in model_name.keys():
                fr = get_map_metrics(P[tag], pgm, yml, out_dir, seq)
                free[tag] = 100*fr
        for tag in model_name.keys():
            # if tag == 'RNIN':
            #     breakpoint()
            rec = {
                'Seq':seq,'Model':tag,
                'FDE':fde(P[tag], G[tag]),#[950:-(G[tag].shape[0] - P[tag].shape[0] - 950), :] if tag == 'RNIN' else G[tag]),
                'ATE':ate(P[tag], G[tag]),#[950:-(G[tag].shape[0] - P[tag].shape[0] - 950), :] if tag == 'RNIN' else G[tag]),
                'RTE':rte(P[tag], G[tag], delta=args.delta_rte, fs=args.fs),#[950:-(G[tag].shape[0] - P[tag].shape[0] - 950), :] if tag == 'RNIN' else G[tag], delta=args.delta_rte, fs=args.fs),
            }
            # quantile metrics only for m1 (others will be NaN)
            if args.make_map_metrics:
                rec.update({'Free%': free[tag]})
            if args.make_calib_metrics:
                rec.update({'PICP':plot_picp[tag][-1], 'AIW':plot_aiw[tag][-1],})
            records.append(rec)
            if args.make_cdf_plot:
                cdf_ate[tag].append(ate(P[tag],G[tag]))
                cdf_rte[tag].append(rte(P[tag],G[tag], delta=args.delta_rte, fs=args.fs))
                cdf_all_ate[tag].append(all_ate(P[tag],G[tag]))
                cdf_all_rte[tag].append(all_rte(P[tag],G[tag], delta=args.delta_rte, fs=args.fs))


    # ---------- DRIFT PLOT  (all models) -----------------------------------


    if args.make_drift_plot:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        colours = dict(UMGLoc='C0', UMGLoc_no_map='C1')     # tweak if needed
        labels   = dict(UMGLoc='UMGLoc (map-aware)',
                        UMGLoc_no_map='UMGLoc (IMU only)')
        for model, curves in drift_store.items():
            # # 1. overlay individual sequences

        
            # 2. median ± IQR summary band on a common grid
            x_grid, matrix = aggregate_curves(curves)        # (Nseq, 501)
            
            # breakpoint()
            # Converting to percentage for the y-axis
               
            min_drift = np.min(matrix, axis=0)
            max_drift = np.max(matrix, axis=0)
            mean_drift = np.mean(matrix, axis=0)
            # std_drift = np.nanstd(matrix, axis=0)

            # breakpoint()


            
            # q5, median, q95 = np.nanpercentile(matrix, [5, 50, 95], axis=0)
            # std = np.std(stack, axis=0)
            # mean = np.mean(stack, axis=0)
            # upp = mean + std
            # low = mean - std
        
            ax.fill_between(x_grid, min_drift, max_drift,
                             color=colours[model], alpha=0.15)
            ax.plot(x_grid, mean_drift,
                     color=colours[model], label=labels[model])
           
 
    
        
        # ax.margins(x=0)
        ax.margins(y=0)
        yt = ax.get_yticks()
        ax.set_ylim(yt[0], yt[-1])   # now the axis ends exactly at the last tick
        pad = 0.02 * (yt[-1] - yt[0])   # 2% of span
        ax.set_ylim(yt[0] + pad, yt[-1])
        yt = np.arange(yt[0], yt[-1]+2.5, 2.5)
        ax.set_yticks(yt[1:])
        # ax.yaxis.set_major_locator(FixedLocator(yt))
        
        ax.legend(ncol=2,           # all items on one row
                   handlelength=2.5,
                   columnspacing=2)
        ax.grid(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        out = out_dir / f"map_drift_plot.pdf"
        fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
        with open(out_dir/f'map_drift_plot.pickle', "wb") as f:
            pickle.dump(fig,f)
        plt.close(fig)



    
    # ---- save table ---------------------------------------------------------
    df = pd.DataFrame.from_records(records)
    print("Rows per model:\n", df.groupby("Model").size())
    # ---- after you build `df` (the per-sequence records) -------------
    
    if args.make_p_test:
        metrics = df.columns[2:]
        pvals = sig_tests(df, metrics, tags[0], tags[1:])                         # <— NEW
        pvals.to_csv(out_dir/'sig_pvals.csv', float_format='%.3e')

    small = summarise_small(df)

    small.to_csv(out_dir/'accuracy_small.csv', float_format='%.3f')
    

    if args.latex:
        # compact main-paper table
        small_tex = small.to_latex(column_format='lcccc',
            caption='Accuracy (mean ±~std). Lower is better.',
            label='tab:acc_small', escape=False)
        (out_dir/'accuracy_small.tex').write_text(small_tex)

    if args.make_calib_metrics:
        # Compute mean and std for each severity
        severities, mean_picp, std_picp = [], [], []
        stats_dict = defaultdict(list)
        for key, values in plot_picp.items():
            model, sev_str = key.split('_perturb')
            severity = float(sev_str)
            stats_dict[model].append((severity, np.mean(values), np.std(values)))
        
        # Prepare plot
        fig, ax = plt.subplots(figsize=FIGSIZE)
        handles = []
        labels = []
        for model, data in stats_dict.items():
            # sort by severity
            data.sort(key=lambda x: x[0])
            sev, mean, std = zip(*data)
            h = ax.plot(sev, mean, marker='o', linestyle='-', color=styles[model], label=model)
            ax.fill_between(sev, np.array(mean) - np.array(std), np.array(mean) + np.array(std), alpha=0.15, color=styles[model])
            handles.append(h[0])
            labels.append(model)
        
        # Reference line
        ax.axhline(0.9, linestyle='--', color='k', label='Nominal 0.9')
        

        ax.set_ylim(0, 1.05)
        ax.grid(True)
        plt.tight_layout()
        out = out_dir / f"picp_plot.pdf"
        fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
        
        fig_leg = plt.figure(figsize=(3.5, 1))   # width x height in inches
        fig_leg.legend(handles, labels,
                   ncol=len(labels),           # all items on one row
                   loc='center',
                   frameon=False,
                   handlelength=2.5,
                   columnspacing=1.2)
        fig_leg.savefig(out_dir / f"{seq}_picp_legend.pdf",
                    bbox_inches='tight',
                    transparent=True)
        
        plt.close(fig)
        plt.close(fig_leg)


        stats_aiw = defaultdict(list)
        for key, values in plot_aiw.items():
            model, sev_str = key.split('_perturb')
            severity = float(sev_str)
            stats_aiw[model].append((severity, np.mean(values), np.std(values)))
        # Prepare plot
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for model, data in stats_aiw.items():
            # sort by severity
            data.sort(key=lambda x: x[0])
            sev, mean, std = zip(*data)
        
            ax.plot(sev, mean, marker='o', linestyle='-', color=styles[model], label=model)
            ax.fill_between(sev, np.array(mean) - np.array(std), np.array(mean) + np.array(std), alpha=0.15, color=styles[model])
        
        # Reference line
        ax.axhline(0.9, linestyle='--', color='gray', label='Nominal 0.9')
        

        plt.tight_layout()
        out = out_dir / f"aiw_plot.pdf"
        fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
        plt.close(fig)

    # ---------- AGGREGATED CDF PLOT  --------------------------------------
    if args.make_cdf_plot:
        fig, ax = plt.subplots(figsize=FIGSIZE)

        handles = []
        labels = []

        # offset = 0.1  # vertical separation between lines
        # idx = 1
        for tag in model_name.keys():
            drift_ate = [i for i in cdf_all_ate[tag]]
            drift_ate = np.sort(np.hstack(drift_ate))
            cdf  = np.arange(0, len(drift_ate)) / (len(drift_ate) - 1)
            h = ax.plot(drift_ate, cdf, color=styles[tag], label=tag) 
            # max_err = np.max(drift_ate)
            # y_max = cdf[np.argmax(drift_ate)]
            # y_shifted = y_max - idx * offset
            # ax.hlines(y_shifted, xmin=min(drift_ate), xmax=max_err, colors='gray', linestyles='dashed')
            # ax.text(max_err, y_shifted, f"{max_err:.2f}", color='gray',
            # va='bottom', ha='right')
        
            handles.append(h[0])
            labels.append(tag)

        ax.set_ylim(-0.02, 1.02)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir/f'cdf_ate.eps',
                    format='eps', bbox_inches='tight', dpi=300)
        with open(out_dir/f'cdf_ate.pickle', "wb") as f:
            pickle.dump(fig,f)

        fig_leg = plt.figure(figsize=(1.5, 0.8))   # width x height in inches
        fig_leg.legend(handles, labels,
                   ncol=len(labels),           # all items on one row
                   loc='center',
                   frameon=False,
                   handlelength=2.5,
                   columnspacing=1.2)
        fig_leg.savefig(out_dir / f"{seq}_cdf_legend.pdf",
                    bbox_inches='tight',
                    transparent=True)
    
        plt.close(fig)
        plt.close(fig_leg)
        
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for tag in tags:
            # breakpoint()
            drift_rte = [i for i in cdf_all_rte[tag]]
            drift_rte = np.sort(np.hstack(drift_rte).squeeze())
            cdf  = np.arange(0, len(drift_rte)) / (len(drift_rte) - 1)
            ax.plot(drift_rte, cdf, color=styles[tag], label=tag)

    
        ax.set_ylim(-0.02, 1.02)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir/f'cdf_rte.eps',
                    format='eps', bbox_inches='tight', dpi=300)
        with open(out_dir/f'cdf_rte.pickle', "wb") as f:
            pickle.dump(fig,f)
        
        plt.close(fig)

    
    # ---------- CALIBRATION FIGURE ---------------------------------------
    # if args.make_calib_plot:
    #     picp_all = {'UMGLoc': [r['PICP'] for r in records if r['Model']=='UMGLoc']}
    #     fig, ax = plt.subplots(figsize=FIGSIZE)
    #     ax.scatter([args.alpha]*len(picp_all['UMGLoc']), picp_all['UMGLoc'],
    #                marker='o', label='Our Model', alpha=.7)
    #     ax.plot([0,1],[0,1],'k--',lw=.5)
    #     ax.set_xlabel('Nominal coverage α')
    #     ax.set_ylabel('Empirical coverage')
    #     ax.set_xlim(0,1); ax.set_ylim(0,1)
    #     ax.legend()
    #     plt.tight_layout()
    #     fig.savefig(out_dir/'calib.eps', format='eps', bbox_inches='tight', dpi=300)
    #     plt.close(fig)

    # ---------- BOX-AND-WHISKER  (ATE & ADE) -------------------------------
        



if __name__ == '__main__':
    main()
