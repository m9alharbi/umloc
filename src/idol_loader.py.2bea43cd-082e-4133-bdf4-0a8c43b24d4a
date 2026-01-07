import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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


class Sequence(CompiledSequence):
    def __init__(self, data_path=None):
        super().__init__()
        self.ts, self.features, self.targets, self.gt_pos = None, None, None, None

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        df = pd.read_csv(data_path + '.csv')
        # duration = (o_df['time_stamp'].values[-1] - o_df['time_stamp'].values[0])/1000
        # num_samples = int(duration * 100)
        # time_axis = np.linspace(o_df['time_stamp'].values[0], o_df['time_stamp'].values[-1], num_samples)
        # inter = interp1d(o_df['time_stamp'].values, o_df.values.T, kind='linear', fill_value="extrapolate")
        # interp_data = inter(time_axis)
        # df = pd.DataFrame(interp_data.T, columns=o_df.columns)
        # df['time_stamp'] = time_axis
        # df = df.interpolate(method='linear')
        start_frame = 0
        end_frame = 1
        self.input_features = ['linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z']
        # self.input_features = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z']
        # self.input_features = ['zed_acceleration_x', 'zed_acceleration_y', 'zed_acceleration_z', 'zed_angular_velocity_x', 'zed_angular_velocity_y', 'zed_angular_velocity_z']
        # self.input_features = ['imu_linear_acc_x', 'imu_linear_acc_y', 'imu_linear_acc_z', 'imu_angular_vel_x', 'imu_angular_vel_y', 'imu_angular_vel_z']
        self.target_features = ['position_x', 'position_y']
        positions = df[self.target_features].values
        self.ts = df['time_stamp'].values / 1000
        # ts_sec = self.ts / 1000
        fs   = 1.0 / np.median(np.diff(self.ts)) #/1000)   # sample rate (Hz)
        fc   = 4.0                                      # cut-off (Hz)
        b, a = butter(N=4, Wn=fc/(0.5*fs), btype='low')
        pos_filt = filtfilt(b, a, positions, axis=0)


        dt = (self.ts[1:] - self.ts[:-1])[:, None]#/1000

        # self.glob_v = np.empty_like(pos_filt[:-2])
        # self.glob_v[:] = (pos_filt[2:] - pos_filt[:-2]) / (ts_sec[2:] - ts_sec[:-2])[:,None]
        # self.glob_v = np.vstack([(pos_filt[1]-pos_filt[0])/(ts_sec[1]-ts_sec[0]), self.glob_v])


        self.glob_v = (positions[1:] - positions[:-1]) / dt
        # norm = np.linalg.norm(self.glob_v, axis=1)
        # good_data = norm < 1.5
        self.ori = ['orientation_qx', 'orientation_qy', 'orientation_qz', 'orientation_qw']
        self.grv = ['game_rotation_mag', 'game_rotation_x', 'game_rotation_y', 'game_rotation_z']
        glob_gyro, glob_acce = self.transform_imu_to_hacf(df[self.input_features[3:]].values, df[self.input_features[:3]].values, df[self.grv].values)
        self.targets = self.glob_v[start_frame:-end_frame]
        self.features = np.concatenate([glob_acce, glob_gyro], axis=1)[start_frame:-end_frame]
        self.ts = self.ts[start_frame:-end_frame]
        # self.features = df[self.input_features][start_frame:-end_frame].values
        self.gt_pos = df[self.target_features][start_frame:-end_frame].values
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