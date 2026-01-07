"""
* This file is part of RNIN-VIO
*
* Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

from scipy.interpolate import interp1d
import pandas
import random
from numpy.random import normal as gen_normal
from os import path as osp
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset
import logging
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from abc import ABC, abstractmethod
import quaternion
from scipy.signal import butter, filtfilt


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """
    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_gt_pos(self):
        pass

    @abstractmethod
    def get_time_stamp(self):
        pass


class SenseINSSequence(CompiledSequence):
    def __init__(self, data_path=None):
        super().__init__()
        (
            self.ts,
            self.features,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (None, None, None, None, None, None)
        self.imu_freq = 100
        self.interval = 1
        self.data_valid = False
        self.sum_dur = 0
        self.valid = False
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        file = osp.join(data_path, 'SenseINS.h5')
        if osp.exists(file):
            imu_all = pandas.read_hdf(file, 'imu_all')
        else:
            file = osp.join(data_path, 'SenseINS.csv')
            if osp.exists(file):
                imu_all = pandas.read_csv(file)
                imu_all.to_hdf(osp.join(data_path, 'SenseINS.h5'), key='imu_all', mode='w')
            else:
                logging.info(f"dataset_fb.py: file is not exist. {file}")
                return

        if 'times' in imu_all:
            tmp_ts = np.array(imu_all[['times']].values)
        else:
            tmp_ts = np.array(imu_all[['time']].values)


        tmp_ts = np.squeeze(tmp_ts)
        tmp_vio_q = np.array(imu_all[['gt_q_w', 'gt_q_x', 'gt_q_y', 'gt_q_z']].values)
        self.get_gt = True
        if tmp_vio_q[0][0] == 1.0 and tmp_vio_q[100][0] == 1.0 or tmp_vio_q[0][0] == tmp_vio_q[-1][0]:
            tmp_vio_q = np.array(imu_all[['vio_q_w', 'vio_q_x', 'vio_q_y', 'vio_q_z']].values)
            tmp_vio_p = np.array(imu_all[['vio_p_x', 'vio_p_y', 'vio_p_z']].values)
            self.get_gt = False
        else:
            tmp_vio_p = np.array(imu_all[['gt_p_x', 'gt_p_y', 'gt_p_z']].values)

        tmp_gyro = np.array(imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values)
        tmp_accel = np.array(imu_all[['acce_x', 'acce_y', 'acce_z']].values)
    
        tmp_vio_gyro_bias = np.array(imu_all[['vio_gyro_bias_x', 'vio_gyro_bias_y', 'vio_gyro_bias_z']].values)
        tmp_vio_acce_bias = np.array(imu_all[['vio_acce_bias_x', 'vio_acce_bias_y', 'vio_acce_bias_z']].values)
        tmp_gyro = tmp_gyro - tmp_vio_gyro_bias[-1, :]
        tmp_acce = tmp_accel - tmp_vio_acce_bias[-1, :]

        start_ts = tmp_ts[10]
        end_ts = tmp_ts[10] + int((tmp_ts[-20]-tmp_ts[1]) * self.imu_freq) / self.imu_freq
        ts = np.arange(start_ts, end_ts, 1.0/self.imu_freq)
        self.data_valid = True
        self.sum_dur = end_ts - start_ts


        vio_q_slerp = Slerp(tmp_ts, Rotation.from_quat(tmp_vio_q[:, [1, 2, 3, 0]]))
        vio_r = vio_q_slerp(ts)
        vio_p = interp1d(tmp_ts, tmp_vio_p, axis=0)(ts)
        gyro = interp1d(tmp_ts, tmp_gyro, axis=0)(ts)
        acce = interp1d(tmp_ts, tmp_acce, axis=0)(ts)

        dt = (ts[self.interval:] - ts[:-self.interval])[:, None]
        ts = ts[:, np.newaxis]


        ori_R_vio = vio_r
        ori_R = ori_R_vio

        gt_disp = (vio_p[self.interval:] - vio_p[: -self.interval])/dt

        glob_gyro = np.einsum("tip,tp->ti", ori_R.as_matrix(), gyro)
        glob_acce = np.einsum("tip,tp->ti", ori_R.as_matrix(), acce)
        glob_acce -= np.array([0.0, 0.0, 9.805])


        self.ts = ts
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)
        self.orientations = ori_R.as_quat()   # [x, y, z, w]
        self.gt_pos = vio_p[:, :2]
        self.gt_ori = ori_R_vio.as_quat()
        self.targets = gt_disp[:, :2]
        print()

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_gt_pos(self):
        return self.gt_pos

    def get_time_stamp(self):
        return self.ts

    def get_aux(self):
        return np.concatenate(
            [self.ts, self.orientations, self.gt_pos], axis=1
        )
