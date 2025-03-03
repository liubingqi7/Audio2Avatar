import os
import cv2
import json
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join
import imageio
import sys
from datasets.freeman_loader import Freeman_Modified
import matplotlib.pyplot as plt

class MultiviewFreeManDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.free_man = Freeman_Modified(args.data_path, fps=args.fps, subject=args.subject, split=args.split)
        self.sessions = self.free_man.get_children_sessions('subj03')
        print(f"Total sessions: {len(self.sessions)}")

        if len(self.sessions) == 0:
            raise ValueError(f"No sessions found for subject {args.subject}")
        
        self.indices = []
        for session in self.sessions:
            video_path = self.free_man.get_video_path(session, cam=1)
            if video_path is None or not os.path.exists(video_path):
                continue
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for frame_idx in range(0, total_frames, args.frame_interval):
                self.indices.append((session, frame_idx))

        print(f"Total samples in FreeManMultiViewDataset: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        session, frame_idx = self.indices[index]
        
        all_views = list(range(1, 9))
        input_views = random.sample(all_views, 4)
        # target_view = random.choice(list(set(all_views) - set(input_views)))
        
        input_imgs = []
        for cam in all_views:
            video_path = self.free_man.get_video_path(session, cam)
            imgs = self.free_man.load_frames(video_path, frame_ids=[frame_idx])
            if imgs is None or len(imgs) == 0:
                raise ValueError(f"Failed to load frame {frame_idx} from session {session}, cam {cam}")
            img = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
            input_imgs.append(img)
        
        # target_video_path = self.free_man.get_video_path(session, target_view)
        # target_imgs = self.free_man.load_frames(target_video_path, frame_ids=[frame_idx])
        # if target_imgs is None or len(target_imgs) == 0:
        #     raise ValueError(f"Failed to load target frame {frame_idx} from session {session}, cam {target_view}")
        # target_img = cv2.cvtColor(target_imgs[0], cv2.COLOR_BGR2RGB)
        
        input_imgs = np.stack(input_imgs, axis=0)  # (4, H, W, 3)
        input_imgs = np.transpose(input_imgs, (0, 3, 1, 2))  # (4, 3, H, W)
        input_imgs = torch.from_numpy(input_imgs.astype(np.float32)) / 255.0
        
        # target_img = np.transpose(target_img, (2, 0, 1))  # (3, H, W)
        # target_img = torch.from_numpy(target_img.astype(np.float32)) / 255.0

        camera_group, cam_params_all = self.free_man.load_camera_group(self.free_man.camera_dir, session)
        input_cam_params = {cam: cam_params_all[cam - 1] for cam in input_views}
        # target_cam_params = cam_params_all[target_view - 1]
        
        # # 加载运动参数（SMPL）
        # # 这里采用其中一个输入视角来加载（假设不同视角下的 SMPL 参数一致）
        # smpl_poses, smpl_scaling, global_orient, smpl_trans, smpl_betas = self.free_man.load_motion(session, input_views[0])
        # # 对应当前帧的参数
        # smpl_pose_frame = smpl_poses[frame_idx]         # (24, 3)
        # global_orient_frame = global_orient[frame_idx]    # (3,)
        # smpl_trans_frame = smpl_trans[frame_idx]          # (3,)
        # # smpl_betas 通常为 (10,) 常量

        sample = {
            'session': session,
            'frame_idx': frame_idx,
            'input_imgs': input_imgs,           # (4, 3, H, W)
            # 'target_img': target_img,           # (3, H, W)
            'input_views': input_views,         # 列表，例如 [2, 5, 6, 8]
            # 'target_view': target_view,
            'input_cam_params': input_cam_params,  # 字典，key 为摄像机索引，value 为参数
            # 'target_cam_params': target_cam_params,
            # 'smpl_params': {
            #     'body_pose': torch.tensor(smpl_pose_frame, dtype=torch.float32),
            #     'global_orient': torch.tensor(global_orient_frame, dtype=torch.float32),
            #     'trans': torch.tensor(smpl_trans_frame, dtype=torch.float32),
            #     'beta': torch.tensor(smpl_betas, dtype=torch.float32)
            # }
        }
        return sample

if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(
        data_path='./data/freeman/wangjiongwow___FreeMan',
        fps=25,
        subject='subj03',
        split='',
        frame_interval=10,
        num_samples=1000
    )
    dataset = MultiviewFreeManDataset(args)
    print("Dataset length:", len(dataset))
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Input images shape:", sample['input_imgs'].shape)

    # show input images
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(sample['input_imgs'][i].permute(1, 2, 0))
        plt.title(f"Input Image {i+1}")
    plt.show()

