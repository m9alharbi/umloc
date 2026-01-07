from utils import *
import numpy as np
import h5py
from os import path as osp
import matplotlib.pyplot as plt
import quaternion

from abc import ABC, abstractmethod

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

    @abstractmethod
    def get_aux(self):
        pass

class NeuritSequence(CompiledSequence):
    """
    Datasets: RIDI, RONIN
    Property: global coordinate frame
    """
    # add 3-axis magnetometer
    feature_dim = 6
    # feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        super().__init__()
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}
        #self.w = kwargs.get('interval', 1)
        #self.magn_status = kwargs.get('use_magnetometer')
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        # raw data: gyroscope (3D), accelerometer (3D), gravity (1D), magnetometer (3D), game vector (4D),
        print("the data_path is:", data_path)
        data = np.load(osp.join(data_path, 'rawdata.npy'))
        ground_truth = np.load(osp.join(data_path, 'groundtruth.npy'))
        # already in global coordinate frame and start from start frame
        # timestamp (1D) gyroscope (3D), accelerometer (3D), magnetometer (3D), rotation vector (4D)
        gyro = data[:, 1:4]
        acce = data[:, 4:7]
        #magn = data[:, 7:10]
        #magn_diff = np.diff(magn, axis=0)
        #magn_diff = np.concatenate([magn_diff, magn_diff[-1, :].reshape(1,3)], axis=0)
        ts = ground_truth[:, 0]
        # tango position
        tango_pos = ground_truth[:, 1:4]
        # tango orientation
        tango_ori = ground_truth[:, 4:8]
        dt = (ts[1:] - ts[:-1])[:, None]
        glob_v = (tango_pos[1:] - tango_pos[:-1]) / dt
        # start_frame = self.info.get('start_frame', 0)
        self.ts = ts

        #if self.magn_status == True:
            #self.features = np.concatenate([gyro, acce, magn_diff], axis = 1)
        #elif self.magn_status == False:
        self.features = np.concatenate([acce, gyro], axis=1)
        #self.magn = magn
        #self.targets = tango_pos[:, :2]#glob_v[:, :2]
        self.targets = glob_v[:, :2]
        self.orientations = tango_ori
        self.gt_pos = tango_pos[:, :2]
        # plt.plot(self.targets)
        # plt.show()
        # plt.hist(self.targets[:, 0], bins=20, edgecolor="white")
        # plt.show()

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_time_stamp(self):
        return self.ts
        
    def get_gt_pos(self):
        return self.gt_pos

    def get_aux(self):
      return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis = 1)

    #def get_magn(self):
        #return self.magn
