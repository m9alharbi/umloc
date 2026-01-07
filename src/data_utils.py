from utils import *
from kaust_loader import Sequence
from ronin_loader import GlobSpeedSequence
from neurit_loader import NeuritSequence
from rnin_loader import SenseINSSequence
from idol_loader import IdolSequence
import numpy as np
import random
from os import path as osp
from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt, rotate, affine_transform
from torch.utils.data import DataLoader, Dataset
import pdb
import quaternion
from PIL import Image
import yaml
from numba import jit
from scipy.spatial.transform import Rotation
from numpy.random import normal as gen_normal
from matplotlib import image


def _coord_transform(world_coords_orig, map_name):
    if map_name == "building1":  # or map_name == 'building1_big':
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 0] = -world_coords_orig[:, 0]
        world_coords_new[:, 1] = world_coords_orig[:, 1]
    elif map_name == "building3":
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 1] = world_coords_orig[:, 0]
        world_coords_new[:, 0] = world_coords_orig[:, 1]
    elif map_name == "building2_f1":
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 0] = world_coords_orig[:, 1]
        world_coords_new[:, 1] = world_coords_orig[:, 0]
    elif map_name == "building2":
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 0] = world_coords_orig[:, 1]
        world_coords_new[:, 1] = world_coords_orig[:, 0]
    return world_coords_new

def _world_to_image_coords(world_coords, meta_info, use_resized=True):
    origin = meta_info['original_origin']
    map_size = meta_info['original_shape']
    world_size = meta_info['world_size']
    scale_x = meta_info['scale_x']
    scale_y = meta_info['scale_y']

    true_x_size = world_size[0]  # m
    true_y_size = world_size[1]  # m
    world_origin_offset_x = origin[0]  # m from top left corner of map
    world_origin_offset_y = origin[1]  # m from top left corner of map
    scale_factor_x = true_x_size/map_size[0]# * scale_x # m/pixel
    scale_factor_y = true_y_size/map_size[1]# * scale_y  # m/pixel

    image_coord_x = np.round(
        ((world_coords[:, 0] + world_origin_offset_x)/scale_factor_x) * scale_y).astype(int)
    image_coord_y = np.round(
        ((world_coords[:, 1] + world_origin_offset_y)/scale_factor_y) * scale_x).astype(int)
    image_coords = np.hstack([image_coord_x[:, None], image_coord_y[:, None]])
    return image_coords
    




def world_to_image_coords(world_coords, meta_info):
    return _world_to_image_coords(world_coords, meta_info)



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
    # breakpoint()
    xy_pix = (traj_world - origin_xy[:, None, :]) / res[:, None, None]
    xy_pix[..., 1] = Ho[:, None] - xy_pix[..., 1]              # flip y

    # -------- original → resized-pixel -------------------------------------
    if use_resized:
        xy_pix[..., 0] = xy_pix[..., 0] * sx[:, None]
        xy_pix[..., 1] = xy_pix[..., 1] * sy[:, None]

    return xy_pix[..., 0], xy_pix[..., 1]
    
def load_sequences(seq_type, data_path_list, map_id_list=None):
    features_all, targets_all, aux_all, maps_all = [], [], [], []
    for i in range(len(data_path_list)):
        seq = seq_type(data_path_list[i])

        feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
        features_all.append(feat)
        targets_all.append(targ)
        aux_all.append(aux)
        if map_id_list is not None:
            maps_all.append(map_id_list[i])  # Add map ID for this sequence

    return features_all, targets_all, aux_all, maps_all

class SequenceDataset(Dataset):
    def __init__(self, seq_type, data_dir, data_list, args, shuffle=True, max_norm=-3.0, transform=None, random_shift=0, map_id_list=None, map_path_dict=None, augment=False):
        super(SequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = args.input_dim
        self.target_dim = args.output_dim
        self.target_type = args.target_type
        self.use_map = args.use_map
        self.grid_size = args.map_size
        self.random_shift = random_shift
        self.augment = augment
        self.accel_bias_range = 0.1
        self.gyro_bias_range = 0.002
        self.feat_acc_sigma = 0.0001
        self.feat_gyr_sigma = 0.00001
        self.gravity_noise_theta_range = 5 #degree
        self.dataset = args.dataset
        self.data_dir = data_dir


        self.window_size = args.window_size
        self.step_size = args.step_size
        self.max_norm = max_norm
        self.transform = transform

        self.data_path_list = [osp.join(data_dir, data) for data in data_list]
        self.map_path_dict = map_path_dict  # file paths, not tensors
        self.map_id_list = map_id_list
        self.index_map = []
        self.features, self.targets, aux, self.maps = load_sequences(seq_type, self.data_path_list, self.map_id_list)

        # Data filtering/smoothing
        if args.feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=args.feat_sigma, axis=0) for feat in self.features]
        if args.targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=args.targ_sigma, axis=0) for targ in self.targets]

        self.ts, self.orientations, self.gt_pos = [], [], []
        for i in range(len(data_list)):
            if args.target_type == 'global_vel':
                self.targets[i] = self.targets[i]
                self.features[i] = self.features[i][:-1]
                self.ts.append(aux[i][:-1, :1])
                self.orientations.append(aux[i][:-1, 1:5])
                self.gt_pos.append(aux[i][:-1, 5:8])

            else:
                self.targets[i] = self.targets[i]
                self.features[i] = self.features[i]
                self.ts.append(aux[i][:, :1])
                self.orientations.append(aux[i][:, 1:5])
                self.gt_pos.append(aux[i][:, 5:8])

            if args.dataset == 'kaust':
                norm = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
                med = np.median(norm)
                mad = np.median(np.abs(norm - med)) + 1e-6
                thr = med + 6.0 * mad
                bad = norm > thr
                good_idx = np.where(~bad)[0]
                for d in range(2):
                    self.targets[i][bad,d] = np.interp(np.where(bad)[0], good_idx, self.targets[i][good_idx,d])
            elif args.dataset == 'idol':
                self.max_norm = -1.0
            else:
                self.max_norm=3.0
        

            if self.max_norm > 0:
                norm = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
                bad_data = norm > self.max_norm
                for j in range(self.window_size + self.random_shift, self.targets[i].shape[0], self.step_size):
                    if not bad_data[j - self.window_size - self.random_shift:j+self.random_shift].any():
                        self.index_map.append([i, j])

            else:
                for j in range(self.window_size + self.random_shift, self.targets[i].shape[0], self.step_size):
                    self.index_map.append([i, j])
        if shuffle:
            random.shuffle(self.index_map)
        
    def load_idol_map(self, yaml_path, gaussian_sigma=1, fixed_size=(64, 64)):
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)

        png_path = metadata['image']
        origin = np.array(metadata['origin'])
        free_thresh = metadata['free_thresh']
        world_size = metadata['world_size']
        map_name = metadata['map_name']
    
        # Load image and normalize
        # Load PGM using PIL
        # img = Image.open(png_path)
        # img = np.array(img).astype(np.float32).T
        img = image.imread(png_path).T
        if img is None:
            raise FileNotFoundError(f"Map image not found: {png_path}")
    
        img[img < free_thresh] = 0.
        img[img >= free_thresh] = 1.
    
        img = 1-img.astype(int)
    
        # # # crop up to the edges of the buildings
        while np.all(img[0, :] == 0):
            img = img[1:, :]
        while np.all(img[-1, :] == 0):
            img = img[:-1, :]
        while np.all(img[:, 0] == 0):
            img = img[:, 1:]
        while np.all(img[:, -1] == 0):
            img = img[:, :-1]
    
        img = 1-img
    
        # blacken everything outside the actual map
        for idx in range(img.shape[0]):
            for jdx in range(img.shape[1]):
                if img[idx, jdx] != 0:
                    img[idx, jdx] = 0
                else:
                    break
            for jdx in range(img.shape[1]-1, 0, -1):
                if img[idx, jdx] != 0:
                    img[idx, jdx] = 0
                else:
                    break
        # img = img.T
        resolution = (world_size/np.array(img.shape)).mean()
        dist_map = self.process_map(img, resolution)
        original_size = dist_map.shape
        resized_map = Image.fromarray(dist_map).resize(fixed_size[::-1], Image.BILINEAR)
        resized_map = np.array(resized_map, dtype=np.float32)

        # plt.imshow(resized_map, cmap='gray')
        # plt.show()

        # Adjust origin proportionally to new size
        scale_x = fixed_size[1] / original_size[1]
        scale_y = fixed_size[0] / original_size[0]
        new_origin = origin.copy()
        new_origin[0] *= scale_x
        new_origin[1] *= scale_y
        meta_info = {
            'resolution': resolution,
            'origin': new_origin,
            'original_origin': origin,
            'original_shape': original_size,
            'free_thresh': free_thresh,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'world_size': world_size,
            'map_name': map_name
        }
        return resized_map.T, meta_info
    
    def load_map(self, yaml_path, gaussian_sigma=1, fixed_size=(64, 64)):
        """
        Loads and processes a binary map:
        - Applies thresholding
        - Morphological filtering
        - Gaussian smoothing
        - Distance transform (normalized)
    
        Args:
            yaml_path (str): Path to .yaml file describing the map
            gaussian_sigma (float): Gaussian smoothing before distance transform
    
        Returns:
            distance_map (np.ndarray): Final [1, H, W] distance map
            meta_info (dict): Map metadata (resolution, origin, etc.)
        """
        # Load metadata
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)

        pgm_path = self.data_dir + metadata['image']
        resolution = metadata['resolution']
        origin = np.array(metadata['origin'])
        negate = metadata['negate']
        free_thresh = metadata['free_thresh']
    
        # Load image and normalize
        # Load PGM using PIL
        img = Image.open(pgm_path)
        img = np.array(img).astype(np.float32)
        if img is None:
            raise FileNotFoundError(f"Map image not found: {pgm_path}")
            
        if negate == 0:
            img = (255 - img) / 255 # ROS convention occ=0 black (>occ_thresh), free=1 (<free_thresh) white 
        else:
            img = img / 255



        # Step 1: Threshold to binary map (0 = obstacle, 1 = free)
        img = img < free_thresh
        dist_map = self.process_map(img, resolution)

        # Resize to fixed size
        original_size = dist_map.shape
        resized_map = Image.fromarray(dist_map).resize(fixed_size[::-1], Image.BILINEAR)
        resized_map = np.array(resized_map, dtype=np.float32)

        # plt.imshow(resized_map, cmap='gray')
        # plt.show()

        # Adjust origin proportionally to new size
        scale_x = fixed_size[1] / original_size[1]
        scale_y = fixed_size[0] / original_size[0]
        new_origin = origin.copy()
        new_origin[0] *= scale_x
        new_origin[1] *= scale_y

    
        # Step 2: Morphological filtering (remove noise, fill gaps)

        # Return results
        meta_info = {
            'resolution': resolution,
            'origin': new_origin,
            'original_origin': origin,
            'original_shape': original_size,
            'free_thresh': free_thresh,
            'negate': negate,
            'scale_x': scale_x,
            'scale_y': scale_y
        }
        return resized_map, meta_info
    
    def make_uniform_map_and_meta(self, gt_traj, grid_size= 64, margin=1.0, distance_value=1.0, resolution=0.05):
        """
        Build a *dummy* distance map that contains only free space (all ones) and a
        meta-dict that mirrors `load_map()` so downstream code does **not** need to
        branch.
    
        Returns
        -------
        distance_map : np.ndarray  shape [1, H, W]  (H=W=grid_size)
        meta         : dict        all the keys your real-map case provides
        """
        # --------------------------------------------------------------------- #
        # 1. Bounding box around the trajectory (+ optional margin)             #
        # --------------------------------------------------------------------- #
        x_min = float(gt_traj[:, 0].min() - margin)
        x_max = float(gt_traj[:, 0].max() + margin)
        y_min = float(gt_traj[:, 1].min() - margin)
        y_max = float(gt_traj[:, 1].max() + margin)
    
        # World extents we need to cover
        extent_x, extent_y = x_max - x_min, y_max - y_min
        extent = max(extent_x, extent_y)              # keep square aspect ratio
    
        # --------------------------------------------------------------------- #
        # 2. Choose a resolution so the *largest* side fits exactly grid_size    #
        # --------------------------------------------------------------------- #
        if resolution is None:                        # metres / pixel
            resolution = extent / (grid_size - 1)
    
        # --------------------------------------------------------------------- #
        # 3. Create the distance map  (all free space)                           #
        # --------------------------------------------------------------------- #
        distance_map = np.full((1, grid_size, grid_size),
                               distance_value, dtype=np.float32)
    
        # --------------------------------------------------------------------- #
        # 4. Compose a meta-dict identical to `load_map()` output                #
        # --------------------------------------------------------------------- #
        # ROS map YAML reports the world-coords of the *bottom-left* pixel.
        origin_bottom_left = np.array([x_min, y_min], dtype=np.float32)
    
        meta = {
            "resolution"      : float(resolution),
            "origin"          : origin_bottom_left.copy(),   # already "resized"
            "original_origin" : origin_bottom_left,          # pre-resize (same here)
            "original_shape"  : (grid_size, grid_size),
            "resized_shape"   : (grid_size, grid_size),
            "scale_x"         : 1.0,                         # no resizing happened
            "scale_y"         : 1.0,
            # the following two fields are *not* used by your maths, but your
            # code expects them to exist – keep them for full compatibility
            "negate"          : 0,
            "free_thresh"     : 0.0,
        }
    
        return distance_map, meta

    def process_map(self, map_array, resolution=0.05):
        """
        Converts a raw binary map into a normalized distance map after Gaussian smoothing.
        Input: map_tensor [1, H, W]
        Output: processed map [1, H, W]
        """
        processed_map = apply_gaussian_smoothing_then_threshold(map_array, sigma=1.0, thresh=0.6)
        processed_map = apply_opening_then_closing(processed_map, 2)
        distance_map = distance_transform_edt(processed_map)
        distance_map = distance_map * resolution
        #distance_map = 1 - (distance_map / distance_map.max())
        # distance_map = distance_map / distance_map.max()
        return distance_map
        
    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))
            
        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])
        ground_truth_pos = np.copy(self.gt_pos[seq_id][frame_id - self.window_size:frame_id])
        initial_pos = np.copy(self.gt_pos[seq_id][frame_id - self.window_size])
        ts = np.copy(self.ts[seq_id][frame_id - self.window_size:frame_id])
        # Per batch computation velocity/displacement

        if self.target_type == 'vel':
            dt = 0.033
            target = (targ[1:, :] - targ[:-1, :]) / dt
            feature = feat[:-1]
        elif self.target_type =='disp':
            #targ = targ - initial_pos
            target = targ[1:, :] - targ[:-1, :]
            feature = feat[:-1]
            gt_pos = ground_truth_pos[:-1]
        else:
            target = targ
            feature = feat
            gt_pos = ground_truth_pos

        if self.use_map:
            # Load the correct map
            map_id = self.maps[seq_id]
            map_path = self.map_path_dict[map_id]
            if self.dataset == 'kaust':
                map_array, map_meta = self.load_map(map_path, fixed_size=(self.grid_size, self.grid_size))
            else:
                map_array, map_meta = self.load_idol_map(map_path, fixed_size=(self.grid_size, self.grid_size))
            if len(map_array.shape[:]) == 2:
                map_array = map_array[None, ::]

        else:

            map_array, map_meta = self.make_uniform_map_and_meta(self.gt_pos[seq_id], grid_size=self.grid_size, margin=1.0)

            # map_array = map_array[None, ::]


        if self.transform is not None:
            map_array, map_meta = self.make_uniform_map_and_meta(self.gt_pos[seq_id], grid_size=self.grid_size, margin=1.0)
            feature, target = self.transform(feature, target) #, map_array)
            
        if self.augment:
            # shift in the accel and gyro bias terms
            random_bias = np.random.random((1, 6))
            random_bias[:, 0:3] = (random_bias[:, 0:3] - 0.5) * self.gyro_bias_range / 0.5
            random_bias[:, 3:6] = (random_bias[:, 3:6] - 0.5) * self.accel_bias_range / 0.5
            feature += random_bias
            
            angle_rand = random.random() * np.pi * 2
            vec_rand = np.array([np.cos(angle_rand), np.sin(angle_rand), 0])
            theta_rand = (random.random() * np.pi * self.gravity_noise_theta_range / 180.0)
            rvec = theta_rand * vec_rand
            r = Rotation.from_rotvec(rvec)
            R_mat = r.as_matrix()
            feature[:, 0:3] = np.matmul(R_mat, feature[:, 0:3].T).T
            feature[:, 3:6] = np.matmul(R_mat, feature[:, 3:6].T).T

            if self.feat_gyr_sigma > 0:
                feature[:, 0:3] += gen_normal(loc=0.0, scale=self.feat_gyr_sigma, size=(len(feature[:, 0]), 3))
            if self.feat_acc_sigma > 0:
                feature[:, 3:6] += gen_normal(loc=0.0, scale=self.feat_acc_sigma, size=(len(feature[:, 0]), 3))


        return feature.astype(np.float32), target.astype(np.float32), initial_pos.astype(np.float32), gt_pos.astype(np.float32), map_array.astype(np.float32), map_meta, ts.astype(np.float32)

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i, gt_pos):
        if self.use_map:
            map_id = self.maps[i]
            map_path = self.map_path_dict[map_id]
            if self.dataset == 'kaust':
                map_array, map_meta = self.load_map(map_path, fixed_size=(self.grid_size, self.grid_size))
            else:
                map_array, map_meta = self.load_idol_map(map_path, fixed_size=(self.grid_size, self.grid_size))
            if len(map_array.shape[:]) == 2:
                map_array = map_array[None, ::]
        else:

            map_array, map_meta = self.make_uniform_map_and_meta(self.gt_pos[i], grid_size=self.grid_size, margin=1.0)

        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32), map_array.astype(np.float32)[np.newaxis,], map_meta

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    """Perform quaternion multiplication."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def rotate_point_with_quaternion(point, q):
    """Rotate a 3D point using a quaternion."""
    p_quat = np.array([0, *point])  # Represent the point as a quaternion
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, p_quat), q_conj)
    return rotated[1:]  # Extract the rotated 3D point (x, y, z)

def dot_product_arr(v1, v2):
    if v1.ndim == 1:
        v1 = np.expand_dims(v1, axis=0)
    if v2.ndim == 1:
        v2 = np.expand_dims(v2, axis=0)
    assert v1.shape[0] == v2.shape[0], '{} {}'.format(v1.shape, v2.shape)
    dp = np.matmul(np.expand_dims(v1, axis=1), np.expand_dims(v2, axis=2))
    return np.squeeze(dp, axis=(1, 2))


def quaternion_from_two_vectors(v1, v2):
    """
    Compute quaternion from two vectors. v1 and v2 need not be normalized.

    :param v1: starting vector
    :param v2: ending vector
    :return Quaternion representation of rotation that rotate v1 to v2.
    """
    one_dim = False
    if v1.ndim == 1:
        v1 = np.expand_dims(v1, axis=0)
        one_dim = True
    if v2.ndim == 1:
        v2 = np.expand_dims(v2, axis=0)
    assert v1.shape == v2.shape
    v1n = v1 / np.linalg.norm(v1, axis=1)[:, None]
    v2n = v2 / np.linalg.norm(v2, axis=1)[:, None]
    w = np.cross(v1n, v2n)
    q = np.concatenate([1.0 + dot_product_arr(v1n, v2n)[:, None], w], axis=1)
    q /= np.linalg.norm(q, axis=1)[:, None]
    if one_dim:
        return q[0]
    return q

def interpolate_quaternion_linear(data, ts_in, ts_out):
    """
    This function interpolate the input quaternion array into another time stemp.

    Args:
        data: Nx4 array containing N quaternions.
        ts_in: input_timestamp- N-sized array containing time stamps for each of the input quaternion.
        ts_out: output_timestamp- M-sized array containing output time stamps.
    Return:
        Mx4 array containing M quaternions.
    """

    assert np.amin(ts_in) <= np.amin(ts_out), 'Input time range must cover output time range'
    assert np.amax(ts_in) >= np.amax(ts_out), 'Input time range must cover output time range'
    pt = np.searchsorted(ts_in, ts_out)
    d_left = quaternion.from_float_array(data[pt - 1])
    d_right = quaternion.from_float_array(data[pt])
    ts_left, ts_right = ts_in[pt - 1], ts_in[pt]
    d_out = quaternion.quaternion_time_series.slerp(d_left, d_right, ts_left, ts_right, ts_out)
    return quaternion.as_float_array(d_out)
    
def align_trajectories_with_quaternion(quat_ref, quat_misaligned, point_ref_first, points_misaligned):
    """
    Align trajectories using numpy-quaternion library for quaternion operations.
    """
    # Convert input quaternions to numpy-quaternion objects
    q_ref = quaternion.from_float_array(quat_ref)
    q_misaligned = quaternion.from_float_array(quat_misaligned)

    # Compute the relative quaternion
    q_relative = q_ref * q_misaligned.conj()

    # Rotate all misaligned points
    p_quat = quaternion.from_float_array(np.concatenate([np.zeros([points_misaligned.shape[0], 1]), points_misaligned], axis=1))
    rotated_points = quaternion.as_float_array(q_relative * p_quat * q_relative.conj())[:, 1:]



    # Compute the translation vector
    translation_vector = point_ref_first - rotated_points[0]

    # Apply the translation to all rotated points
    aligned_points = rotated_points + translation_vector
    return aligned_points, q_relative, translation_vector


    
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R

def align_trajectories(quat_ref, quat_misaligned, point_ref, point_misaligned, points_misaligned):
    # Convert quaternions to rotation matrices
    R_ref = quaternion_to_rotation_matrix(quat_ref)
    R_misaligned = quaternion_to_rotation_matrix(quat_misaligned)
    
    # Compute relative rotation
    R_relative = R_ref @ np.linalg.inv(R_misaligned)
    
    # Compute translation vector
    T = point_ref - R_relative @ point_misaligned
    
    # Align trajectory
    aligned_points = (R_relative @ points_misaligned.T).T + T
    return aligned_points, R_relative, T
    """
    ref_points = dataset.gt_pos[0]
    misaligned_points = dataset.gt_pos[1]
    point_ref = ref_points[0]
    point_misaligned = misaligned_points[0]
    quat_ref = [1, 0, 0, 0]  # Identity quaternion (no rotation)
    quat_misaligned = [0.866, 0, 0, 0.5]  # 30° rotation around Z-axis
    aligned_points, R, T = align_trajectories(quat_ref, quat_misaligned, point_ref, point_misaligned, misaligned_points)
    """
def get_map_data(gt_pos, sigma, grid_size, map_type='all'):

    all_traj_data = np.concatenate(gt_pos)
    x_min, x_max = all_traj_data[:, 0].min() - 1, all_traj_data[:, 0].max() + 1
    y_min, y_max = all_traj_data[:, 1].min() - 1, all_traj_data[:, 1].max() + 1
    feasibility = np.zeros((grid_size, grid_size))
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    if map_type == 'loaded':
        feasibility = np.ones(grid_size)
    elif map_type == 'trajectory':
        for trajectory in gt_pos:
            for point in trajectory:
                ix = np.argmin(np.abs(x - point[0]))
                iy = np.argmin(np.abs(y - point[1]))
                feasibility[iy, ix] = 1  # Mark the point as free 
    else:
        feasibility = np.ones((grid_size, grid_size))
    feasibility = gaussian_filter(feasibility, sigma=1)
    distance_map = distance_transform_edt(feasibility)
    #distance_map = 1 - (distance_map / distance_map.max())
    distance_map = distance_map / distance_map.max()

    #distance_map = np.exp(-distance_map**2 / (2 * 2**2))
    feasibility = torch.tensor(feasibility, dtype=torch.float32, device='cuda')
    distance_map = torch.tensor(distance_map, dtype=torch.float32, device='cuda')
    return distance_map, (x_min, x_max, y_min, y_max)

#@jit
def change_cf(ori, vectors):
    """
    Euler-Rodrigous formula v'=v+2s(rxv)+2rx(rxv)
    :param ori: quaternion [n]x4
    :param vectors: vector nx3
    :return: rotated vector nx3
    """
    assert ori.shape[-1] == 4
    assert vectors.shape[-1] == 3

    if len(ori.shape) == 1:
        ori = np.repeat([ori], vectors.shape[0], axis=0)
    q_s = ori[:, :1]
    q_r = ori[:, 1:]

    tmp = np.cross(q_r, vectors)
    vectors = np.add(np.add(vectors, np.multiply(2, np.multiply(q_s, tmp))), np.multiply(2, np.cross(q_r, tmp)))
    return vectors

class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, feat, targ): #, map_img=None, meta=None):
        # if map_img is not None:
            # for t in self.transforms:
                # feat, targ, map_img = t(feat, targ, map_img, meta)
            # return feat, targ, map_img
        # else:
        for t in self.transforms:
            feat, targ = t(feat, targ)
        return feat, targ



class RandomHoriRotateSeq:
    def __init__(self, input_format, output_format=None):
        """
        Rotate global input, global output by a random angle
        @:param input format - input feature vector(x,3) boundaries as array (E.g [0,3,6] or [0,3,6,9])
        @:param output format - output feature vector(x,2/3) boundaries as array (E.g [0,2,5])
                                if 2, 0 is appended as z.
        """
        self.i_f = input_format
        self.o_f = output_format

    #@jit
    def __call__(self, feature, target):
        a = np.random.random() * 2 * np.math.pi
        # print("Rotating by {} degrees", a/np.math.pi * 180)
        t = np.array([np.cos(a), 0, 0, np.sin(a)])

        for i in range(len(self.i_f) - 1):
            feature[:, self.i_f[i]: self.i_f[i + 1]] = \
                change_cf(t, feature[:, self.i_f[i]: self.i_f[i + 1]])

        for i in range(len(self.o_f) - 1):
            if self.o_f[i + 1] - self.o_f[i] == 3:
                vector = target[:, self.o_f[i]: self.o_f[i + 1]]
                target[:, self.o_f[i]: self.o_f[i + 1]] = change_cf(t, vector)
            elif self.o_f[i + 1] - self.o_f[i] == 2:
                vector = np.concatenate([target[:, self.o_f[i]: self.o_f[i + 1]], np.zeros([target.shape[0], 1])],
                                        axis=1)
                target[:, self.o_f[i]: self.o_f[i + 1]] = change_cf(t, vector)[:, :2]


        return feature, target

class SensorPerturb:
    def __init__(self,
                 noise_std_acc =0.02,
                 noise_std_gyro=0.003,
                 dropout_p=0.10,
                 dropout_len=30):
        self.acc_std  = noise_std_acc
        self.gyro_std = noise_std_gyro
        self.p_drop   = dropout_p
        self.L_drop   = dropout_len

    def __call__(self, feature, target, map_img=None):
        imu = feature.copy()             # feature already WITHOUT timestamps
        T, D = imu.shape
        # white Gaussian noise
        noise = np.random.randn(T, D).astype(np.float32)
        noise[:, :3] *= self.acc_std
        noise[:, 3:] *= self.gyro_std
        imu += noise
        # contiguous dropout window
        if np.random.rand() < self.p_drop:
            s = np.random.randint(0, max(1, T - self.L_drop))
            imu[s:s+self.L_drop] = 0.0
        if map_img is not None:
            return imu, target, map_img
        else:
            return feature, target



def get_dataset(data_dir, data_list, args, map_id_list=None, map_path_dict=None, mode=None):

    transforms = []
    input_format = [0, 3, 6]
    output_format = [0, 2]
    random_shift, shuffle, transforms, grv_only, augment = 0, False, [], False, False


    if args.dataset == 'ronin':
        seq = GlobSpeedSequence
        # random shift
        # random rotation
        map_id_list = None
        map_path_dict = None
    elif args.dataset == 'rnin':
        seq = SenseINSSequence
        map_id_list = None
        map_path_dict = None
    elif args.dataset == 'idol':
        if not args.use_map:
            map_id_list = None
            map_path_dict = None
        seq = IdolSequence
    else:
        if not args.use_map:
            map_id_list = None
            map_path_dict = None
        seq = Sequence
        #if args.add_noise:
        #transforms.append(SensorPerturb())
        #transforms.append(rotate_about_center_padded(input_format, output_format))

    if mode == 'test':
        shuffle = False
        grv_only = True
        augment = False
        transforms = []
        random_shift = 0
    elif mode == 'train':
        shuffle = True
        augment = True
        random_shift = args.step_size // 2
        transforms.append(RandomHoriRotateSeq(input_format, output_format))
    elif mode == 'val':
        shuffle = True

    transforms = ComposeTransform(transforms)
    dataset = SequenceDataset(seq, data_dir, data_list, args, shuffle=shuffle, transform=transforms, random_shift=random_shift, map_id_list=map_id_list, map_path_dict=map_path_dict, augment=augment)
    return dataset
def get_data_list(data_dir, list_path, args, mode=None):
    """
    Reads the list of trajectories and automatically generates:
      - data_list (trajectory names)
      - map_id_list (map IDs matching trajectory names)
      - map_data_paths_dict (map ID -> map file path)
    
    Args:
        data_dir: directory containing the trajectory CSVs
        list_path: path to the file listing all trajectories
        args: arguments (optional, for future use)
        maps_dir: directory containing the map files

    Returns:
        data_list: list of trajectory names
        map_id_list: list of corresponding map IDs
        map_data_paths_dict: dictionary {map_id: map_path}
    """
    data_list = []
    map_id_list = []
    map_path_dict = {}

    with open(list_path) as f:
        lines = [s.strip() for s in f.readlines() if len(s) > 0 and s[0] != '#']

    for line in lines:
        data_name = line.split(',')[0]  # get the trajectory file name (without extension)
        data_list.append(data_name)
        if args.use_map:
            # Assume map ID = trajectory name (e.g., trajectory1.csv -> trajectory1_map.pth)
            map_id = data_name
            map_id_list.append(map_id)
            # Build map file path
            map_file_name = f"{map_id}.yaml"  # or .pt depending on your saved map format
            map_file_path = osp.join(data_dir, map_file_name)

            #if not osp.exists(map_file_path):
                #raise FileNotFoundError(f"Map file does not exist: {map_file_path}")

            map_path_dict[map_id] = map_file_path

    return get_dataset(data_dir, data_list, args, map_id_list, map_path_dict, mode)
