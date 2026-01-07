import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from pathlib import Path

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


class IdolSequence(CompiledSequence):
    def __init__(self, data_path=None):
        super().__init__()
        self.ts, self.features, self.targets, self.gt_pos = None, None, None, None

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        df = pd.read_feather(data_path + '.feather')
    
        building_name = Path(data_path).stem[:9]
        
        start_frame = 0
        end_frame = 1
        self.input_features = ['iphoneAccX', 'iphoneAccY', 'iphoneAccZ', 'iphoneGyroX', 'iphoneGyroY', 'iphoneGyroZ']
        # self.input_features = ['stencilAccX', 'stencilAccY', 'stencilAccZ', 'stencilGyroX', 'stencilGyroY', 'stencilGyroZ']
        self.target_features = ['processedPosX', 'processedPosY']
        positions = df[self.target_features].values

    
        if building_name == 'building1':
            # theta = 0.2822
            theta = 0.0 #1.8510
            x = df['processedPosX']*np.cos(theta) - df['processedPosY']*np.sin(theta)
            # positions[:, 1] = positions[:, 0]*np.sin(theta) + positions[:, 1]*np.cos(theta)
            y = df['processedPosX']*np.sin(theta) + df['processedPosY']*np.cos(theta)
            x = -x

        else:
            theta = 1.8510
            # positions[:, 0] = positions[:, 0]*np.cos(theta) - positions[:, 1]*np.sin(theta)
            y = df['processedPosX']*np.cos(theta) - df['processedPosY']*np.sin(theta)
            # positions[:, 1] = positions[:, 0]*np.sin(theta) + positions[:, 1]*np.cos(theta)
            x = df['processedPosX']*np.sin(theta) + df['processedPosY']*np.cos(theta)

        positions = np.stack([x, y], axis=1)
        # plt.plot(positions[:, 0], positions[:, 1])
        # plt.show()
        self.ts = df['timestamp'].values
        # ts_sec = self.ts / 1000

        dt = (self.ts[1:] - self.ts[:-1])[:, None]#/1000


        self.glob_v = (positions[1:] - positions[:-1]) / dt
        # norm = np.linalg.norm(self.glob_v, axis=1)
        # good_data = norm < 1.5
        self.ori = ['orientX', 'orientY', 'orientZ', 'orientW']
        # self.grv = ['game_rotation_mag', 'game_rotation_x', 'game_rotation_y', 'game_rotation_z']
        # glob_gyro, glob_acce = self.transform_imu_to_hacf(df[self.input_features[3:]].values, df[self.input_features[:3]].values, df[self.grv].values)
        self.targets = self.glob_v[start_frame:-end_frame]
        # self.features = np.concatenate([glob_acce, glob_gyro], axis=1)[start_frame:-end_frame]
        self.ts = self.ts[start_frame:-end_frame]
        self.features = df[self.input_features][start_frame:-end_frame].values
        self.gt_pos = positions[start_frame:-end_frame]
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        # ax1.plot(self.gt_pos)
        # ax2.plot(self.targets)
        # ax3.plot(self.features)
        # plt.show()
        # breakpoint()
        self.orientations = df[self.ori][start_frame:-end_frame].values
        # plt.plot(self.targets)
        # plt.show()
        # plt.hist(self.targets[:, 0], bins=20, edgecolor="white")
        # plt.show()

    def transform_imu_to_hacf(self, gyro, acc, game_rv_quats):
        """
        Transforms raw IMU data (gyro, acc) to a gravity-aligned frame using Game Rotation Vector.
    
        Args:
            gyro: [N, 3] angular velocity in device local frame
            acc: [N, 3] linear acceleration in device local frame
            game_rv_quats: [N, 4] quaternion [w, x, y, z] from Android Game Rotation Vector
    
        Returns:
            gyro_hacf: [N, 3] gyro rotated to HACF (gravity-aligned frame)
            acc_hacf: [N, 3] acc rotated to HACF (gravity-aligned frame)
        """
        assert gyro.shape == acc.shape == (game_rv_quats.shape[0], 3)
    
        # Convert to quaternion array
        q_array = quaternion.from_float_array(game_rv_quats)
    
        # Convert gyro and acc vectors to pure quaternions
        gyro_q = quaternion.from_float_array(np.concatenate(
            [np.zeros((gyro.shape[0], 1)), gyro], axis=1
        ))
        acc_q = quaternion.from_float_array(np.concatenate(
            [np.zeros((acc.shape[0], 1)), acc], axis=1
        ))
    
        # Rotate vectors: q * v * q_conj
        gyro_rotated = q_array * gyro_q * q_array.conj()
        acc_rotated = q_array * acc_q * q_array.conj()
    
        # Extract rotated vectors
        gyro_hacf = quaternion.as_float_array(gyro_rotated)[:, 1:]
        acc_hacf = quaternion.as_float_array(acc_rotated)[:, 1:]
    
        return gyro_hacf, acc_hacf
    
    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_gt_pos(self):
        return self.gt_pos

    def get_time_stamp(self):
        return self.ts

    def get_aux(self):
      return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis = 1)