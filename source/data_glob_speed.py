import json
import random
from os import path as osp

import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset
from pyquaternion import Quaternion

from data_utils import CompiledSequence, select_orientation_source, load_cached_sequences


class GlobSpeedSequence(CompiledSequence):
    """
    Dataset :- RoNIN (can be downloaded from http://ronin.cs.sfu.ca/)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1)
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        with open(osp.join(data_path, 'info.json')) as f:
            self.info = json.load(f)

        self.info['path'] = osp.split(data_path)[-1]

        self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source(
            data_path, self.max_ori_error, self.grv_only)

        with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
            gyro_uncalib = f['synced/gyro_uncalib']  #for angular velocity
            # print("Gyro uncalib: ",gyro_uncalib)
            acce_uncalib = f['synced/acce']  # for accelaration
            # print("acce_uncalib: ",acce_uncalib)
            gyro = gyro_uncalib - np.array(self.info['imu_init_gyro_bias'])
            # print("gyro calibrated: ",gyro)
            acce = np.array(self.info['imu_acce_scale']) * (acce_uncalib - np.array(self.info['imu_acce_bias']))
            # print("accelaration_calibrated: ",acce)
            ts = np.copy(f['synced/time'])
            # print("timeseries: ",ts)
            tango_pos = np.copy(f['pose/tango_pos'])
            # print("tango_pos: ",tango_pos)
            init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])  #first
            # print("initial_tango_orientation: ",init_tango_ori)

        #Compute the IMU orientation in the Tango coordinate frame. #shanmu - may be global
        print("------------Compute the IMU orientation in the Tango coordinate frame------------------------")
        # print("ori: ",ori)
        ori_q = quaternion.from_float_array(ori)
        # print(type(ori_q))
        # print("orietation_quaternion: ",ori_q)
        rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration'])
        # print("Rot_imu_to_tango_quaternion",rot_imu_to_tango)
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
        # print("initial_rotor: ",init_rotor)
        ori_q = init_rotor * ori_q
        # print("orientation_quaternion: ",ori_q)

        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        # print("dt: ",dt)

        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt
        print(len(ori_q))
        print(len(glob_v))
        glob_v_q=quaternion.from_float_array(np.concatenate([np.zeros([glob_v.shape[0], 1]), glob_v], axis=1))
        glob_v_contrastive=quaternion.as_float_array(ori_q[200:] * glob_v_q * ori_q[200:].conj())[:, 1:]
        # print("global_velocity: ", glob_v)

        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        # print("gyro_quaternion: ",gyro_q)
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        # print("acce_quaterion: ",acce_q)
        # q1=quaternion.from_float_array([0.369969745324723, 0.629673216173061, 0.363760369952464, -0.578197723777012])
        q1 = Quaternion(axis=[0, 0, 1], angle=3.14159265 / 2)
        print(q1)
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]

        glob_gyro_contrastive=[q1.rotate(l1) for l1 in glob_gyro]
        # glob_gyro_contrastive=quaternion.as_float_array(q1*ori_q*gyro_q*ori_q.conj()*q1.conj())[:,1:]

        # print("global_gyro: ",glob_gyro)
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
        glob_acce_contrastive=[q1.rotate(l2) for l2 in glob_acce]
        # glob_acce_contrastive=quaternion.as_float_array(q1*ori_q * acce_q * ori_q.conj()*q1.conj())[:,1:]
        # print("global_accelation: ",glob_acce)
        start_frame = self.info.get('start_frame', 0)
        self.ts = ts[start_frame:]
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
        self.features_contrastive=np.concatenate([glob_gyro_contrastive,glob_acce_contrastive],axis=1)[start_frame:]

        # zz=0
        # if (zz!=2):
        #     for i in range (len(self.features)):
        #         print([self.features[i],self.features_contrastive[i]])
        #         z=2
        # print("-------features: ",self.features)
        self.targets = glob_v[start_frame:, :2]
        self.targets_contrastive=glob_v_contrastive[start_frame:,:2]
        # print("--------global_targets: ",self.targets)
        self.orientations = quaternion.as_float_array(ori_q)[start_frame:]
        self.gt_pos = tango_pos[start_frame:]

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_features_contrastive(self):
        return self.features_contrastive

    def get_targets_contrastive(self):
        return self.targets_contrastive

    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: device: {}, ori_error ({}): {:.3f}'.format(
            self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])


class DenseSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super().__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=1, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(window_size, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class StridedSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(StridedSequenceDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size)

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        self.features, self.targets,self.features_contrastive,self.targets_contrastive,aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.interval, **kwargs)
        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]
        feat_contrastive=self.features_contrastive[seq_id][frame_id:frame_id + self.window_size]
        targ_contrastive = self.targets_contrastive[seq_id][frame_id]

        # if self.transform is not None:
        #     feat, targ,feat_contrastive,targ_contrastive = self.transform(feat, targ,feat_contrastive,targ_contrastive)

        return feat.astype(np.float32).T, targ.astype(np.float32),feat_contrastive.astype(np.float32).T,targ_contrastive.astype(np.float32) ,seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class SequenceToSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=100, window_size=400,
                 random_shift=0, transform=None, **kwargs):
        super(SequenceToSequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        max_norm = kwargs.get('max_velocity_norm', 3.0)
        self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []
        for i in range(len(data_list)):
            self.features[i] = self.features[i][:-1]
            self.targets[i] = self.targets[i]
            self.ts.append(aux[i][:-1, :1])
            self.orientations.append(aux[i][:-1, 1:5])
            self.gt_pos.append(aux[i][:-1, 5:8])

            velocity = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
            bad_data = velocity > max_norm
            for j in range(window_size + random_shift, self.targets[i].shape[0], step_size):
                if not bad_data[j - window_size - random_shift:j + random_shift].any():
                    self.index_map.append([i, j])

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32), targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32)
