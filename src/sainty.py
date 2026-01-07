##########################################################
# sanity_checks.py
# -- run:  python sanity_checks.py
##########################################################
import torch, torch.nn.functional as F
import numpy as np

def real2pix(
    traj_world,          # tensor/ndarray [..., 2]  (x, y) in metres
    meta,                # dict with required fields
    use_resized=True,    # True → coords on resized map, False → original
    as_tensor=True,      # output torch tensor (True) or NumPy array (False)
):
    """
    Convert real-world (x,y) → pixel coords on the chosen map grid.

    meta must contain:
        'original_origin'  : (2,)  origin of map in world frame
        'resolution'       : float metres / pixel  (original map)
        'original_shape'   : (H_o, W_o)
        'scale_x', 'scale_y'
        'resized_shape'    : (H_r, W_r)  (only used if use_resized=True)
    """
    xy = traj_world.clone()
    device = traj_world.device

    origin = meta['original_origin'][:2]
    origin = torch.tensor(origin, dtype=torch.float32, device=device)
    res    = meta['resolution']

    # 1. world → original-pixel
    xy_pix = (xy - origin) / res          # (...,2)
    # flip y because image origin top-left
    H_o = meta['original_shape'][0]
    xy_pix[..., 1] = H_o - xy_pix[..., 1]

    if use_resized:
        # 2. scale to resized grid
        sx, sy = meta['scale_x'], meta['scale_y']
        xy_pix[..., 0] *= sx
        xy_pix[..., 1] *= sy

    return xy_pix[..., 0], xy_pix[..., 1]

def bilinear_values_torch(dist_map, x_pix, y_pix, map_shape, border_mode="border"):
    """
    dist_map : [1,1,H,W]   distance / feasibility map
    x_pix    : [B,L]       pixel x-coords (float) on SAME map
    y_pix    : [B,L]       pixel y-coords (float)
    map_shape: (H,W)       size of dist_map  (so we don’t rely on .shape inside grad)
    
    Returns  [B,L]  sampled values via bilinear interpolation.
    
    * border_mode = "border":   coords beyond the image are clamped to edge pixels
                   = "zeros" : values outside map set to 0
    """
    H, W = map_shape
    B, L = x_pix.shape
    device = dist_map.device

    # 1.  four neighbouring integer indices
    x0 = torch.floor(x_pix).to(torch.long)
    y0 = torch.floor(y_pix).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    if border_mode == "border":
        x0 = torch.clamp(x0, 0, W - 1)
        x1 = torch.clamp(x1, 0, W - 1)
        y0 = torch.clamp(y0, 0, H - 1)
        y1 = torch.clamp(y1, 0, H - 1)
    else:  # zeros
        mask_out = (
            (x_pix < 0) | (x_pix > W - 1) |
            (y_pix < 0) | (y_pix > H - 1)
        )

    # 2.  gather the four pixel values  (broadcast indexing)
    Ia = dist_map[0, 0, y0, x0]   # [B,L]
    Ib = dist_map[0, 0, y1, x0]
    Ic = dist_map[0, 0, y0, x1]
    Id = dist_map[0, 0, y1, x1]

    # 3.  bilinear weights
    wa = (x1.float() - x_pix) * (y1.float() - y_pix)
    wb = (x1.float() - x_pix) * (y_pix - y0.float())
    wc = (x_pix - x0.float()) * (y1.float() - y_pix)
    wd = (x_pix - x0.float()) * (y_pix - y0.float())

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # [B,L]

    # 4.  optional zero outside
    if border_mode == "zeros":
        out = out.masked_fill_(mask_out, 0.0)

    return out

def aux_attention_loss(attn_weights, pooled_mask, reduction='mean', eps=1e-6):
    """
    Encourage attention to concentrate on free cells.

    Args
    ----
    attn_weights : [B, 1, F]  -- output from nn.MultiheadAttention (softmax done)
    pooled_mask  : [B, 1, H_r/4, W_r/4]  -- after the two MaxPools
                    0 = free , 1 = obstacle
    reduction    : 'mean' or 'sum'

    Returns
    -------
    loss : scalar tensor
    """
    B, _, Hk, Wk = pooled_mask.shape
    F_tokens = Hk * Wk

    free = (pooled_mask == 0).float().view(B, 1, F_tokens)    # [B,1,F]
    free_target = free / (free.sum(dim=-1, keepdim=True) + eps)

    # Binary cross-entropy between current attention and ideal free-only distribution
    loss = F.binary_cross_entropy(attn_weights, free_target, reduction=reduction)
    return loss

def get_collision_rate(
    traj_world,          # [B,L,2] metres  (torch tensor)
    dist_map_full,       # [1,1,H_o,W_o]  distance==0 or occupancy==1 on obstacles
    meta,                # dict with original_shape etc.
    collision_thresh=1e-1   # threshold in map units (0 for exact walls)
):
    """
    Returns scalar collision rate ∈ [0,1].
    """
    device = dist_map_full.device
    B, L, _ = traj_world.shape
    H_o, W_o = meta['original_shape']

    # 1. world -> ORIGINAL pixel coords
    x_grid, y_grid = real2pix(traj_world, meta,
                          use_resized=False, as_tensor=True)

    # 2. bilinear sample via helper
    dist_vals = bilinear_values_torch(dist_map_full,
                                      x_grid,
                                      y_grid,
                                      map_shape=dist_map_full.shape[-2:])     # [B,L]

    collided = (dist_vals <= collision_thresh).float()  # 1 if collision
    return collided.mean()

def get_distance_score(
    traj_world,
    dist_map_small,      # [1,1,H_r,W_r]  distance in **pixels**
    meta,
    reduction='mean'     # 'mean' or 'min'
):
    """
    Returns average (or min) distance-to-obstacle in **metres**.
    Assumes dist_map_small encodes Euclidean pixel distances from walls.
    """
    device = dist_map_small.device
    B, L, _ = traj_world.shape
    H_r, W_r = dist_map_small.shape[-2:]
    scale_factor = meta['resolution'] * (meta['original_shape'][0] / H_r)  # m per pixel in small map

    # 1. world -> RESIZED pixel coords
    x_grid, y_grid = real2pix(traj_world, meta,
                                use_resized=True, as_tensor=True)

    # 2. bilinear sample
    dists_px = bilinear_values_torch(dist_map_small,
                                     x_grid,
                                     y_grid,
                                     map_shape=dist_map_small.shape[-2:])      # [B,L]

    dists_m = dists_px * scale_factor                          # convert to metres

    if reduction == 'mean':
        return dists_m.mean()
    elif reduction == 'min':
        return dists_m.min(dim=1)[0].mean()          # avg min-clearance per traj
    else:
        raise ValueError("reduction must be 'mean' or 'min'")
        
##############################################################
#  sanity_light.py  – checks helpers in isolation
##############################################################

# ---------------------------------------------------------------------
#  build a 6 m × 6 m toy map (6×6 px original, 3×3 px resized)
# ---------------------------------------------------------------------
H_o = W_o = 6
H_r = W_r = 3
RES  = 1.0                     # 1 px == 1 m

origin = np.array([0., 0.])
occ_full = torch.zeros(1, 1, H_o, W_o)   # free = 0
occ_full[:, :, 2:4, 2:4] = 1             # 2×2 block obstacle
dist_full  = 1.0 - occ_full              # free = 1, obstacle = 0
dist_small = F.avg_pool2d(dist_full, 2)  # simple down-sample

meta = {
    'original_origin': origin,
    'resolution'     : RES,
    'original_shape' : (H_o, W_o),
    'scale_x'        : W_r / W_o,
    'scale_y'        : H_r / H_o,
    'resized_shape'  : (H_r, W_r)
}

traj_world = torch.tensor([[[1.0, 1.0],      # definitely free
                            [2.5, 2.5],      # centre of obstacle
                            [4.9, 4.9],      # free
                            [2.1, 3.9]]])    # just inside obstacle edge
# shape [B=1, L=4, 2]


# ---------------------------------------------------------------------
# 1. real2pix
# ---------------------------------------------------------------------
x_pix_o, y_pix_o = real2pix(traj_world.clone(), meta, use_resized=False)
assert torch.isclose(x_pix_o[0, 0], torch.tensor(1.0)),        "real2pix X fail"
assert torch.isclose(y_pix_o[0, 0], torch.tensor(H_o-1.0)),    "real2pix Y flip fail"
print("real2pix ✔  PASS")

# # ---------------------------------------------------------------------
# # 2. bilinear_values_torch  vs  nearest-pixel reference
# # ---------------------------------------------------------------------
with torch.no_grad():
    ref_vals = dist_full[0,0, y_pix_o.long(), x_pix_o.long()]   # nearest px
    bil_vals = bilinear_values_torch(dist_full, x_pix_o, y_pix_o,
                                     map_shape=(H_o, W_o))[0]
    assert torch.allclose(ref_vals, bil_vals, atol=1e-4), "bilinear mismatch"
print("bilinear_values_torch ✔  PASS")

# # ---------------------------------------------------------------------
# # 3. collision-rate  (compare with reference count)
# # ---------------------------------------------------------------------
with torch.no_grad():
    ref_collide = (ref_vals <= 0.1).float().mean()
    test_collide = get_collision_rate(traj_world, dist_full, meta,
                                      collision_thresh=0.1)
    assert abs(ref_collide - test_collide) < 1e-5, "collision-rate mismatch"
print("collision_rate ✔  PASS")

# ---------------------------------------------------------------------
# 4. distance-score  (mean and min must be > 0, mean ≥ min)
# ---------------------------------------------------------------------
d_mean = get_distance_score(traj_world, dist_small, meta, reduction='mean')
d_min  = get_distance_score(traj_world, dist_small, meta, reduction='min')
assert d_min > 0 and d_mean >= d_min, "distance-score logic fail"
print(f"distance_score  mean={d_mean:.3f} m  min={d_min:.3f} m ✔  PASS")

# ---------------------------------------------------------------------
# 5. aux_attention_loss  (loss_free < loss_obstacle)
# ---------------------------------------------------------------------
pooled_mask = F.max_pool2d(occ_full, 2)                     # [1,1,3,3]
F_tok = pooled_mask.numel()
attn_free = torch.zeros(1, 1, F_tok);  attn_free[0,0,0] = 1  # put mass on a free token
attn_obs  = torch.zeros_like(attn_free); attn_obs [0,0,4] = 1
loss_free = aux_attention_loss(attn_free, pooled_mask)
loss_obs  = aux_attention_loss(attn_obs , pooled_mask)
assert loss_free < loss_obs, "aux-loss should prefer free space"
print("aux_attention_loss ✔  PASS")

print("\nAll sanity checks passed 🎉")
##############################################################

