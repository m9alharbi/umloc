import argparse
import json
import os
import sys
from os import path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from scipy.interpolate import interp1d
import math

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from utils import *
from losses import *

def adjust_angle_array(angles):
    """
    Resolve ambiguities within a array of angles. It assumes neighboring angles should be close.
    Args:
        angles: an array of angles.

    Return:
        Adjusted angle array.
    """
    new_angle = np.copy(angles)
    angle_diff = angles[1:] - angles[:-1]

    diff_cand = angle_diff[:, None] - np.array([-math.pi * 4, -math.pi * 2, 0, math.pi * 2, math.pi * 4])
    min_id = np.argmin(np.abs(diff_cand), axis=1)

    diffs = np.choose(min_id, diff_cand.T)
    new_angle[1:] = np.cumsum(diffs) + new_angle[0]
    return new_angle
    
def orientation_to_angles(ori):
    """
    Covert an array of quaternions to an array of Euler angles. Calculations are from Android source code:

    https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/hardware/SensorManager.java
    Function "getOrientation(float[] R, float[] values)"

    Note that this function DOES NOT consider singular configurations, such as Gimbal Lock.

    Args:
        ori: an array of N quaternions.

    Returns:
        A Nx3 array. With Android's game rotation vector or rotation vector, each group of three values
        correspond to: azimuth(yaw), pitch and roll, respectively.
    """
    if ori.dtype != quaternion.quaternion:
        ori = quaternion.from_float_array(ori)

    rm = quaternion.as_rotation_matrix(ori)
    angles = np.zeros([ori.shape[0], 3])
    angles[:, 0] = adjust_angle_array(np.arctan2(rm[:, 0, 1], rm[:, 1, 1]))
    angles[:, 1] = adjust_angle_array(np.arcsin(-rm[:, 2, 1]))
    angles[:, 2] = adjust_angle_array(np.arctan2(-rm[:, 2, 0], rm[:, 2, 2]))

    return angles

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

def icp_fit_transformation(source, target):
    """
    This function computes the best rigid transformation between two point sets. It assumes that "source" and
    "target" are with the same length and "source[i]" corresponds to "target[i]".

    :param source: Nxd array.
    :param target: Nxd array.
    :return: A transformation as (d+1)x(d+1) matrix; the rotation part as a dxd matrix and the translation
    part as a dx1 vector.
    """
    assert source.shape == target.shape
    center_source = np.mean(source, axis=0)
    center_target = np.mean(target, axis=0)
    m = source.shape[1]
    source_zeromean = source - center_source
    target_zeromean = target - center_target
    W = np.dot(source_zeromean.T, target_zeromean)
    U, S, Vt = np.linalg.svd(W)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = center_target.T - np.dot(R, center_source.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--align_length', type=int, default=600)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--stride', type=float, default=0.67)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--show_plot', action='store_true')

    args = parser.parse_args()

    root_dir, data_list = '', []
    if args.path is not None:
        if args.path[-1] == '/':
            args.path = args.path[:-1]
        root_dir = osp.split(args.path)[0]
        data_list = [osp.split(args.path)[1]]
    elif args.list is not None:
        root_dir = args.root_dir
        with open(args.list) as f:
            data_list = [s.strip().split()[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    ate_all, rte_all = [], []
    pred_per_min = 200 * 60

    for data in data_list:
        data_path = osp.join(root_dir, data)
        with open(osp.join(data_path, 'info.json')) as f:
            info = json.load(f)
            device = info['device']
            rot_imu_to_tango = quaternion.quaternion(*info['start_calibration'])
            ref_time_imu = info['imu_reference_time']
            imu_time_offset = info['imu_time_offset']
            start_frame = info.get('start_frame', 0)

        with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
            ori = np.copy(f['synced/game_rv'])
            tango_ori_q = quaternion.from_float_array(f['pose/tango_ori'])
            ori_q = quaternion.from_float_array(ori)
            init_rotor = tango_ori_q[0] * rot_imu_to_tango * ori_q[0].conj()
            ori = quaternion.as_float_array(init_rotor * ori_q)[start_frame:]

            ts = np.copy(f['synced/time'])[start_frame:]
            tango_pos = np.copy(f['pose/tango_pos'])[start_frame:]
            step = np.copy(f['raw/imu/step'])
        step_ts = step[:, 0] / 1e09 - imu_time_offset
        sid, eid = 0, step_ts.shape[0] - 1
        while sid < step_ts.shape[0] and ts[0] > step_ts[sid]:
            sid += 1
        while eid >= 0 and ts[-1] < step_ts[eid]:
            eid -= 1
        assert sid < step_ts.shape[0] and eid >= 0
        breakpoint()

        step_ts = step_ts[sid:eid + 1]
        step = step[sid:eid + 1]

        ori_at_step = interpolate_quaternion_linear(ori, ts, step_ts)
        yaw_at_step = orientation_to_angles(ori_at_step)[:, 0]

        rot_hori = np.stack([np.cos(yaw_at_step), np.sin(yaw_at_step),
                             -np.sin(yaw_at_step), np.cos(yaw_at_step)], axis=1).reshape([-1, 2, 2])
        step_v = np.stack([np.zeros(step.shape[0]), np.ones(step.shape[0]) * args.stride], axis=1)

        step_glob = np.squeeze(np.matmul(rot_hori, np.expand_dims(step_v, axis=2)), axis=2)
        pos_pred = np.zeros([step_glob.shape[0] + 1, 2])
        step_ts = np.concatenate([[ts[0]], step_ts], axis=0)
        pos_pred[0] = tango_pos[0][:2]
        pos_pred[1:] = np.cumsum(step_glob, axis=0) + pos_pred[0]

        eid = ts.shape[0] - 1
        while eid >= 0 and ts[eid] > step_ts[-1]:
            eid -= 1
        pos_gt = tango_pos[:eid + 1, :2]
        pos_pred = interp1d(step_ts, pos_pred, axis=0)(ts[:eid + 1])

        if args.align_length is not None and args.align_length > 0:
            _, r, t = icp_fit_transformation(pos_pred[:args.align_length], pos_gt[:args.align_length])
            pos_pred = np.matmul(r, pos_pred.T).T + t

        # For trajectories shorted than 1 min, we scale the RTE value accordingly.
        if pos_pred.shape[0] < pred_per_min:
            ratio = pred_per_min / pos_pred.shape[0]
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pos_pred.shape[0] - 1) * ratio
        else:
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pred_per_min)
        rte_all.append(rte)

        ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
        ate_all.append(ate)
        print('Sequence {}, ate {:.6f}, rte {:.6f}'.format(data, ate, rte))

        plt.close('all')
        plt.figure(data, figsize=(8, 6))
        plt.plot(pos_gt[:, 0], pos_gt[:, 1])
        plt.plot(pos_pred[:, 0], pos_pred[:, 1])
        plt.legend(['Ground truth', 'Estimated'])
        plt.axis('equal')
        plt.tight_layout()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            plt.savefig(osp.join(args.out_dir, data + '_pdr.png'))

        if args.show_plot:
            plt.show()

    print('All done. Average ate:{:.6f}, average rte: {:.6f}'.format(np.mean(ate_all), np.mean(rte_all)))
    if args.out_dir is not None and osp.isdir(args.out_dir):
        with open(osp.join(args.out_dir, 'result.csv'), 'w') as f:
            f.write('seq,ate,rte\n')
            for i in range(len(data_list)):
                f.write('{},{:.6f},{:.6f}\n'.format(data_list[i], ate_all[i], rte_all[i]))