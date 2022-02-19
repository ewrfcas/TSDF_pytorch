from torch.utils.data import Dataset
from utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import pickle


class NormalizedMVSDTU(Dataset):
    def __init__(self, root_dir, pred_depth_dir, scan, img_wh=None, max_len=-1, max_len_per=-1):
        self.root_dir = root_dir
        with open('data/norm_param.pkl', 'rb') as f:
            meta = pickle.load(f)
            self.norm_mat = meta['norm_mat']
            self.norm_scale = meta['scale']
            self.near = meta['near']
            self.far = meta['far']

        if img_wh is None:
            self.img_wh = [640, 480]
        else:
            self.img_wh = img_wh
        self.scale_factor = 1.0 / 200
        self.max_len = max_len
        self.max_len_per = max_len_per
        self.scan = scan
        self.pred_depth_dir = pred_depth_dir

        self.build_metas()
        self.build_proj_mats()

    def build_metas(self):
        self.metas = []
        self.scans = [self.scan]
        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3]
        for scan in self.scans:
            num_viewpoint = 49
            if self.max_len_per > 0:
                split_idx = num_viewpoint // self.max_len_per
                valid_view_idx = np.arange(0, num_viewpoint, step=split_idx)[:self.max_len_per]
            else:
                valid_view_idx = np.arange(0, num_viewpoint)

            # viewpoints (49)
            for i in range(num_viewpoint):
                if i in valid_view_idx:
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, i)]

    def load_K_Rt_from_P(self, P):
        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()  # w2c-->c2w !!!!!!!!!!!!!!!!!
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return intrinsics, pose

    def build_proj_mats(self):
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        for vid in range(49):
            proj_mat_filename = os.path.join(self.root_dir, f'Cameras/{vid:08d}_cam.txt')
            intrinsic, extrinsic, _ = self.read_cam_file(proj_mat_filename)
            x_scale = self.img_wh[0] / 1600
            y_scale = self.img_wh[1] / 1200
            intrinsic[0, :] *= x_scale
            intrinsic[1, :] *= y_scale

            extrinsic[:3, 3] *= self.scale_factor  # scale for the offsets
            K = np.eye(4)
            K[:3, :3] = intrinsic
            P = K @ extrinsic @ self.norm_mat
            P = P[:3, :4]
            intrinsic, c2w = self.load_K_Rt_from_P(P)
            intrinsic = intrinsic[:3, :3]
            w2c = np.linalg.inv(c2w)

            intrinsics += [intrinsic.copy()]
            world2cams += [w2c.copy()]
            cam2worlds += [c2w.copy()]

        self.intrinsics = np.stack(intrinsics)
        self.world2cams = np.stack(world2cams)
        self.cam2worlds = np.stack(cam2worlds)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
        self.depth_interval = float(lines[11].split()[1])
        return intrinsics, extrinsics, [depth_min, depth_max]

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (1200, 1600)
        depth_h = cv2.resize(depth_h, self.img_wh, interpolation=cv2.INTER_NEAREST)

        return depth_h

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, vid = self.metas[idx]

        # NOTE that the id in image file names is from 1 to 49 (not 0~48)
        img_filename = os.path.join(self.root_dir, f'DTU_IDR/dtu_{scan}/image/{vid:06d}.png')
        mask_filename = os.path.join(self.root_dir, f'DTU_IDR/dtu_{scan}/mask/{vid:03d}.png')
        depth_filename = os.path.join(self.root_dir, f'Depths_raw/{scan}/depth_map_{vid:04d}.pfm')
        manual_mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)  # [1200,1600]
        manual_mask[manual_mask <= 127.5] = 0
        manual_mask[manual_mask > 127.5] = 1
        manual_mask = cv2.resize(manual_mask, self.img_wh, interpolation=cv2.INTER_AREA)  # (512,640)
        manual_mask[manual_mask > 0] = 1

        img = Image.open(img_filename)
        img = img.resize(self.img_wh, Image.BILINEAR)

        depth_h = self.read_depth(depth_filename)
        depth_h *= self.scale_factor
        depth_h /= self.norm_scale
        if self.pred_depth_dir is not None:
            depth_pred = self.read_depth(os.path.join(self.pred_depth_dir,
                                                      f'{scan}/depth_est/{vid:08d}.pfm'))
            depth_pred *= self.scale_factor
            depth_pred /= self.norm_scale
        else:
            depth_pred = np.zeros((1, 1))

        sample['images'] = np.array(img)
        sample['gt_depths'] = depth_h.astype(np.float32)
        sample['depths_pred'] = depth_pred.astype(np.float32)
        sample['intrinsics'] = self.intrinsics[vid].astype(np.float32)
        sample['w2cs'] = self.world2cams[vid].astype(np.float32)
        sample['c2ws'] = self.cam2worlds[vid].astype(np.float32)
        sample['view_id'] = vid
        sample['scan'] = scan
        sample['manual_masks'] = manual_mask.astype(np.float32)

        return sample
