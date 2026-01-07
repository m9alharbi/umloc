"""
ATE evaluator (RMSE + MAE) for 2-D trajectories.

Folder layout
-------------
a058_3_gt.npy          # ground-truth
a058_3_umgloc.npy      # UMGLoc prediction
...
<idx>_rnin.npy         # RNIN outputs, where <idx> is 0…N-1
"""

import pathlib, re, sys
import numpy as np
from collections import defaultdict
import pdb


"""
ATE evaluator (2-D) for multiple trajectory files.

File layout
-----------
  <seqID>_gt.npy          # ground-truth, shape (N,2)
  <seqID>_umgloc.npy      # UMGLoc prediction, shape (N,2)
  <idx>_rnin.npy          # RNIN prediction, shape (M,3) – first 2 columns are (x,y)

RNIN files (0_rnin.npy, 1_rnin.npy, …) are matched to GT/UMGLoc sequences
strictly by order after sorting: the 1st GT pairs with 0_rnin, the 2nd with 1_rnin, etc.
"""

# ---------- configuration ----------
ROOT = pathlib.Path("output/")     # change if needed
GT_SUFFIX   = "_gt"
UML_SUFFIX  = "_umgloc"
RNI_SUFFIX  = "_rnin"
# -----------------------------------

def load_xy(path):
    """Load .npy and keep the first two columns."""
    return np.load(path)[:, :2]

def ate_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray):
    """
    Returns:
        ate_rmse : root-mean-square ATE  (√mean ‖e‖²)
        ate_mean : mean absolute ATE     (mean ‖e‖)
    Clips/pads prediction to ground-truth length if necessary.
    """
    n = len(gt_xy)
    if len(pred_xy) >= n:
        pred = pred_xy[:n]
    else:
        pad = np.repeat(pred_xy[-1][None, :], n - len(pred_xy), axis=0)
        pred = np.vstack([pred_xy, pad])

    err = np.linalg.norm(pred - gt_xy, axis=1)   # per-sample Euclidean error
    return float(np.sqrt(np.mean((pred - gt_xy) ** 2))), float(np.mean(err))

def main():
    # -------- gather file lists --------
    gt_files  = sorted(ROOT.glob(f"*{GT_SUFFIX}.npy"))      # ground truth
    uml_map   = {f.stem.replace(UML_SUFFIX, ""): f          # UMGLoc lookup
                 for f in ROOT.glob(f"*{UML_SUFFIX}.npy")}
    rni_files = sorted(                                    # RNIN numeric order
        ROOT.glob(f"*{RNI_SUFFIX}.npy"),
        key=lambda p: int(re.match(r"(\d+)", p.stem).group(1))
    )

    if not gt_files:
        sys.exit("No *_gt.npy files found in the directory.")

    overall = defaultdict(list)   # accumulate per-model RMSEs

    # -------- per-sequence evaluation --------
    for idx, gt_path in enumerate(gt_files):
        seq_id = gt_path.stem.replace(GT_SUFFIX, "")
        gt = load_xy(gt_path)

        print(f"\nSequence: {seq_id:<15}  len={len(gt)}")

        # UMGLoc
        if seq_id in uml_map:
            uml = load_xy(uml_map[seq_id])
            ate_rmse, ate_mean = ate_metrics(uml, gt)
            overall["umgloc"].append(ate_rmse)
            print(f"  UMGLoc   ATE-RMSE={ate_rmse:7.3f} m   "
                  f"ATE-mean={ate_mean:7.3f} m")
        else:
            print("  UMGLoc   (file missing)")

        # RNIN (ordered matching)
        if idx < len(rni_files):
            rni = load_xy(rni_files[idx])
            ate_rmse, ate_mean = ate_metrics(rni, gt)
            overall["rnin"].append(ate_rmse)
            print(f"  RNIN     ATE-RMSE={ate_rmse:7.3f} m   "
                  f"ATE-mean={ate_mean:7.3f} m   "
                  f"(from {rni_files[idx].name})")
        else:
            print("  RNIN     (no RNIN file left)")

    # -------- aggregate summary --------
    print("\n--- Aggregate ATE-RMSE over all sequences ---")
    for tag in ["umgloc", "rnin"]:
        if overall[tag]:
            print(f"  {tag:<7}: mean ATE-RMSE = "
                  f"{np.mean(overall[tag]):7.3f} m  "
                  f"(across {len(overall[tag])} sequences)")

if __name__ == "__main__":
    main()
