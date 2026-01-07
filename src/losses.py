import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
#TODO: evaluation metric (ATE, RTE)

def pinball_loss(targ, pred, q):
    error = targ - pred
    loss = torch.max(q * error, (q - 1) * error)
    return loss

def combined_pinball_loss(pred, targ, quantiles):
    loss = 0
    # for i, q in enumerate(quantiles):
    #     quantile_loss = pinball_loss(targ, pred[:, :, i], q)
    #     loss += quantile_loss
    losses = [pinball_loss(targ, pred[..., i], q)
              for i, q in enumerate(quantiles)]

    loss = torch.stack(losses, dim=-1).mean(-1)  # (B, T)
    return loss

class QTrajLoss(torch.nn.Module):
    def __init__(self, quantile, args, dts):
        """
        Calculate position loss in global coordinate frame using pinball loss for quantile regression
        """
        super(QTrajLoss, self).__init__()
        self.quantile = quantile
        self.target_type = args.target_type
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.dt = dts
        self.target_iqr = 0.5
        self.λ_iqr = 0.5


    def forward(self, pred_1, pred_2, targ):
        if self.target_type == 'global_vel':
            pred_pos_1 = torch.cumsum(pred_1[:, 1:, :], dim=1)
            pred_pos_2 = torch.cumsum(pred_2[:, 1:, :], dim=1)
            gt_pos = torch.cumsum(targ[..., 1:, :], dim=1)
            traj_loss_1 = combined_pinball_loss(pred_pos_1, gt_pos[:, :, 0], self.quantile)
            traj_loss_2 = combined_pinball_loss(pred_pos_2, gt_pos[:, :, -1], self.quantile)
            traj_loss = (traj_loss_1.mean() + traj_loss_2.mean())/2
            
            # q_lo = torch.cat((pred_pos_1[:, :, 0], pred_pos_2[:, :, 0]), dim=-1)
            # q_hi = torch.cat((pred_pos_1[:, :, -1], pred_pos_2[:, :, -1]), dim=-1)        # assuming axis order [low,mid,high]
            # spread = q_hi - q_lo                              # [B,T]
            # var_loss = F.mse_loss(spread, self.target_iqr * torch.ones_like(spread))

            # loss = traj_loss + self.λ_iqr * var_loss

            # traj_loss = torch.stack((traj_loss_1, traj_loss_2), dim=-1)
            # traj_loss = torch.mean(traj_loss)
            # pred_pos = torch.cumsum(pred[:, 1:, :], dim=1)
            # gt_pos = torch.cumsum(targ[:, 1:, :], dim=1)
            #traj_loss = obstacle_loss.unsqueeze(-1) * self.mse_loss(pred_pos, gt_pos)
            # traj_loss = self.mse_loss(pred_pos, gt_pos)
            # traj_loss = torch.mean(traj_loss)
            
        else:
            traj_loss_1 = combined_pinball_loss(pred_1, targ[:, :, 0], self.quantile)
            traj_loss_2 = combined_pinball_loss(pred_2, targ[:, :, -1], self.quantile)
            traj_loss = torch.stack((traj_loss_1, traj_loss_2), dim=-1)
            traj_loss = torch.mean(torch.cumsum(traj_loss, dim=1))

        return traj_loss


class TrajLoss(torch.nn.Module):
    def __init__(self, args):
        """
        Calculate position loss in global coordinate frame
        Target :- Global Velocity
        Prediction :- Global Velocity
        """
        super(TrajLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.vel_loss = torch.nn.MSELoss(reduction='none')
        self.alpha = 0.3
        self.target_type = args.target_type

    def forward(self, pred, targ):#, obstacle_loss):
        if self.target_type == 'global_vel':
            # loss1 = (pred - targ).pow(2)
            pred_pos = torch.cumsum(pred, dim=1)
            gt_pos = torch.cumsum(targ, dim=1)
            vel_loss = self.vel_loss(pred, targ)
            # loss2 = (pred_pos - gt_pos).pow(2)
            # loss = torch.cat((loss1, loss2), 1)
            #traj_loss = obstacle_loss.unsqueeze(-1) * self.mse_loss(pred_pos, gt_pos)
            traj_loss = self.mse_loss(pred_pos, gt_pos)
            traj_loss = torch.mean(traj_loss)
            traj_loss = torch.mean(self.alpha * traj_loss + ((1 - self.alpha) * vel_loss))
            # traj_loss = torch.mean(traj_loss + vel_loss)

        else:
            traj_loss = self.mse_loss(pred, targ)
            traj_loss = torch.mean(torch.cumsum(traj_loss, dim=1))
        return traj_loss


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


# ---- BCE with proper label smoothing --------------------------------------
# def gan_g_loss(scores_fake, eps=0.05):
#     # smooth positive labels in [1-eps, 1]
#     y_fake = torch.ones_like(scores_fake) * (1 - random.uniform(0, eps))
#     return bce_loss(scores_fake, y_fake)

# def gan_d_loss(scores_real, scores_fake, eps=0.05):
#     y_real = torch.ones_like(scores_real) * (1 - random.uniform(0, eps))
#     y_fake = torch.zeros_like(scores_fake) + random.uniform(0, eps)
#     return bce_loss(scores_real, y_real) + bce_loss(scores_fake, y_fake)

def gan_g_loss(scores_fake, eps=0.05):
    y = 1 - eps * torch.rand_like(scores_fake)
    return bce_loss(scores_fake, y)

def gan_d_loss(scores_real, scores_fake, eps=0.05):
    y_r = 1 - eps * torch.rand_like(scores_real)
    y_f =      eps * torch.rand_like(scores_fake)
    return bce_loss(scores_real, y_r) + bce_loss(scores_fake, y_f)


def real2pix(
    traj_world: torch.Tensor,   # [B,L,2]  (B==1 at test time)
    meta: dict,                 # batched-tensors in training, scalars in test
    *,
    use_resized: bool = True,
    as_tensor: bool = True,
    training: bool = False,     # NEW → True during back-prop phase
):
    """
    Convert real-world (x,y) → pixel coords.
    In *training*: every key in meta is already a tensor shaped [B,…].
    In *testing* : each key is a scalar / tuple, so we broadcast to B.
    """
    B, L, _ = traj_world.shape
    device  = traj_world.device
    dtype   = traj_world.dtype
    def _to_tensor(v, shape_tail=()):
        """Helper to cast / broadcast meta values."""
        if torch.is_tensor(v):
            return v.to(device=device, dtype=dtype)
        # scalar / tuple → tensor and broadcast to batch
        t = torch.as_tensor(v, dtype=dtype, device=device)
        t = t.expand(B, *shape_tail) if training else t.unsqueeze(0).expand(B, *shape_tail)
        return t

    # -------- load meta (batched or broadcast) ------------------------------
    origin_xy = _to_tensor(meta["original_origin"][...,:2], (2,))      # [B,2]
    res       = _to_tensor(meta["resolution"])                 # [B]
    Ho        = _to_tensor(meta["original_shape"][0])          # [B]
    if use_resized:
        sx    = _to_tensor(meta["scale_x"])                    # [B]
        sy    = _to_tensor(meta["scale_y"])                    # [B]

    # -------- world → original-pixel ---------------------------------------
    xy_pix = (traj_world - origin_xy[:, None, :]) / res[:, None, None]
    xy_pix[..., 1] = Ho[:, None] - xy_pix[..., 1]              # flip y

    # -------- original → resized-pixel -------------------------------------
    if use_resized:
        xy_pix[..., 0] = xy_pix[..., 0] * sx[:, None]
        xy_pix[..., 1] = xy_pix[..., 1] * sy[:, None]

    return xy_pix[..., 0], xy_pix[..., 1]

def _world_to_image_coords(world_coords, meta_info, use_resized=True, training=False):

    B, L, _ = world_coords.shape
    dtype   = world_coords.dtype
    device  = world_coords.device

    def _to_tensor(v, shape_tail=()):
        """Helper to cast / broadcast meta values."""
        if torch.is_tensor(v):
            return v.to(device=device, dtype=dtype)
        # scalar / tuple → tensor and broadcast to batch
        t = torch.as_tensor(v, dtype=dtype, device=device)
        t = t.expand(B, *shape_tail) if training else t.unsqueeze(0).expand(B, *shape_tail)
        return t
    origin = _to_tensor(meta_info['original_origin'], (2,))
    map_size_h = _to_tensor(meta_info['original_shape'][0])
    map_size_w = _to_tensor(meta_info['original_shape'][1])

    world_size_h = _to_tensor(meta_info['world_size'][0])
    world_size_w = _to_tensor(meta_info['world_size'][1])
    scale_x = _to_tensor(meta_info['scale_x'])
    scale_y = _to_tensor(meta_info['scale_y'])
    true_x_size = world_size_h  # m
    true_y_size = world_size_w  # m
    world_origin_offset_x = origin[..., 0]  # m from top left corner of map
    world_origin_offset_y = origin[..., 1]  # m from top left corner of map
    scale_factor_x = true_x_size/map_size_h# * scale_x # m/pixel
    scale_factor_y = true_y_size/map_size_w# * scale_y  # m/pixel

    image_coord_x = ((world_coords[..., 0] + world_origin_offset_x[:, None])/scale_factor_x[:, None]) * scale_y[:, None]
    image_coord_y = ((world_coords[..., 1] + world_origin_offset_y[:, None])/scale_factor_y[:, None]) * scale_x[:, None]

    return image_coord_x, image_coord_y



def bilinear_values_torch(
    dist_map: torch.Tensor,     # [B,1,H,W] in training or [1,1,H,W] in test
    x_grid: torch.Tensor,       # [B,L]
    y_grid: torch.Tensor,       # [B,L]
    map_shape: tuple[int, int],
    *,
    training: bool = False,
    align_corners: bool = True,
):
    """
    Bilinear sampling that works whether dist_map is per-batch or shared.
    """
    B, L = x_grid.shape
    H, W = map_shape

    # If dist_map only has one copy but B>1, expand a view (no memory cost)
    if (not training) and dist_map.size(0) == 1 and B > 1:
        dist_map = dist_map.expand(B, -1, -1, -1)

    # pixel → [-1,1]
    x_norm = x_grid / (W - 1) * 2 - 1
    y_norm = y_grid / (H - 1) * 2 - 1
    grid   = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(1)   # [B,1,L,2]

    vals = F.grid_sample(
        dist_map, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )                                                             # [B,1,1,L]
    return vals.view(B, L)


def feasibility_aware_loss(traj_world,
    dist_map_small,      # [1,1,H_r,W_r]  distance in **pixels**
    meta, safety_m=0.3, margin_m=0.1):
    device = dist_map_small.device
    B, L, _ = traj_world.shape
    H_r, W_r = dist_map_small.shape[-2:]
    # alpha = 2.0
    # scale_factor = meta['resolution'] * (meta['original_shape'][0] / H_r)  # m per pixel in small map

    # 1. world -> RESIZED pixel coords
    if "map_name" in meta:
        x_grid, y_grid = _world_to_image_coords(traj_world, meta,
                                use_resized=True, training=True)
    else:
        x_grid, y_grid = real2pix(traj_world, meta,
                                use_resized=True, as_tensor=True, training=True)

    dists_px = bilinear_values_torch(dist_map_small,
                                     x_grid,
                                     y_grid,
                                     map_shape=dist_map_small.shape[-2:], training=True)      # [B,L]
    # 3) barrier for collisions  (d < r ⇒ heavy penalty)
    # barrier_dist_neg = safety_m - dists_px
    # barrier_loss = torch.exp(alpha * barrier_dist_neg)
    # barrier_loss = barrier_loss.mean()

    # margin_dist_neg = (safety_m + margin_m) - dists_px
    # margin_loss = torch.exp(alpha * margin_dist_neg)
    # margin_loss = margin_loss.mean()

    # # Combine the losses
    # loss = barrier_loss + margin_loss

    barrier = F.relu(safety_m - dists_px)           # 0 if safe, >0 if inside radius
    barrier_loss = (barrier / safety_m)#.pow(2)     # smooth quadratic

    # 4) margin reward  (encourage > r+margin, but fade to 0 afterwards)
    margin = F.relu((safety_m + margin_m) - dists_px)

    margin_loss = (margin / margin_m)#.pow(2)

    loss = barrier_loss.pow(2) + margin_loss.pow(2)
    return loss.mean()

        
def get_collision_rate(
    traj_world,          # [B,L,2] metres  (torch tensor)
    dist_map_full,       # [1,1,H_o,W_o]  distance==0 or occupancy==1 on obstacles
    meta,                # dict with original_shape etc.
    collision_thresh=0.3,   # threshold in map units (0 for exact walls)
    training=False
):
    """
    Returns scalar collision rate ∈ [0,1].
    """
    device = dist_map_full.device
    B, L, _ = traj_world.shape
    H_o, W_o = meta['original_shape']

    # 1. world -> ORIGINAL pixel coords
    if "map_naem" in meta:
        x_grid, y_grid = _world_to_image_coords(traj_world, meta,
                                use_resized=True, training=training)
    else:
        x_grid, y_grid = real2pix(traj_world, meta,
                          use_resized=True, as_tensor=True, training=training)
    # 2. bilinear sample via helper
    dist_vals = bilinear_values_torch(dist_map_full,
                                      x_grid,
                                      y_grid,
                                      map_shape=dist_map_full.shape[-2:], training=training)     # [B,L]

    collided = (dist_vals <= collision_thresh).float()  # 1 if collision
    return torch.mean(collided, dim=1)

    
def get_distance_score(
    traj_world,
    dist_map_small,      # [1,1,H_r,W_r]  distance in **pixels**
    meta,
    reduction='mean',     # 'mean' or 'min'
    training=False
):
    """
    Returns average (or min) distance-to-obstacle in **metres**.
    Assumes dist_map_small encodes Euclidean pixel distances from walls.
    """
    device = dist_map_small.device
    B, L, _ = traj_world.shape
    H_r, W_r = dist_map_small.shape[-2:]
    # scale_factor = meta['resolution'] * (meta['original_shape'][0] / H_r)  # m per pixel in small map

    # 1. world -> RESIZED pixel coords
    if "map_name" in meta:
        x_grid, y_grid = _world_to_image_coords(traj_world, meta,
                                use_resized=True, training=training)
    else:
        x_grid, y_grid = real2pix(traj_world, meta,
                                use_resized=True, as_tensor=True, training=training)

    # 2. bilinear sample
    dists_px = bilinear_values_torch(dist_map_small,
                                     x_grid,
                                     y_grid,
                                     map_shape=dist_map_small.shape[-2:], training=training)      # [B,L]

    #dists_m = dists_px * scale_factor                          # convert to metres
    if reduction == 'mean':
        return torch.mean(dists_px, dim=1)
    elif reduction == 'min':
        return dists_px.min(dim=1)[0].mean()          # avg min-clearance per traj
    else:
        raise ValueError("reduction must be 'mean' or 'min'")

def absolute_trajectory_error(pred_traj, pred_traj_gt):
    loss = torch.sqrt(torch.mean((torch.norm(pred_traj - pred_traj_gt, dim=2)) ** 2, dim=1))
    # loss = torch.sqrt(torch.mean((pred_traj - pred_traj_gt) ** 2))
    # loss = error.mean(dim=1)
    return loss

def final_displacement_error(pred_pos, pred_pos_gt):
    position_drift = torch.norm(pred_pos_gt[:, -1, :] - pred_pos[:, -1, :], dim=-1)
    delta_position = pred_pos_gt[:, 1:, :] - pred_pos_gt[:, :-1, :]
    delta_length = torch.norm(delta_position, dim=-1)
    moving_len = torch.sum(delta_length, dim=1)
    return position_drift / moving_len

def compute_absolute_trajectory_error(est, gt):
    """
    The Absolute Trajectory Error (ATE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: estimated trajectory
        gt: ground truth trajectory. It must have the same shape as est.

    Return:
        Absolution trajectory error, which is the Root Mean Squared Error between
        two trajectories.
    """
    return np.sqrt(np.mean((est - gt) ** 2))


def compute_relative_trajectory_error(est, gt, delta, max_delta=-1):
    """
    The Relative Trajectory Error (RTE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """
    if max_delta == -1:
        max_delta = est.shape[0]
    deltas = np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[deltas[i]:] + gt[:-deltas[i]] - est[:-deltas[i]] - gt[deltas[i]:]
        rtes[i] = np.sqrt(np.mean(err ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(rtes)


def compute_ate_rte(est, gt, pred_per_min=1800):
    """
    A convenient function to compute ATE and RTE. For sequences shorter than pred_per_min, it computes end sequence
    drift and scales the number accordingly.
    """
    ate = compute_absolute_trajectory_error(est, gt)
    if est.shape[0] < pred_per_min:
        ratio = pred_per_min / est.shape[0]
        rte = compute_relative_trajectory_error(est, gt, delta=est.shape[0] - 1) * ratio
    else:
        rte = compute_relative_trajectory_error(est, gt, delta=pred_per_min)

    return ate, rte