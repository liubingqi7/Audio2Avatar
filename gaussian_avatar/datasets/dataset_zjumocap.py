import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils.data_utils import VideoData
from smplx import SMPL
import torch

scene_list = [
    '377',
    '386',
    '387',
    '390',
    '392',
    '393',
    '394'
]

# scene_list = [
#     '377',
# ]

precomputed_pelvis_joints = {
    '377': np.array([-0.00170379, -0.22081676, 0.02813518]),
    '386': np.array([-0.00167512, -0.22100767, 0.02874197]),
    '387': np.array([-0.0017419, -0.22306314, 0.02913309]),
    '390': np.array([-0.00156097, -0.21801242, 0.02844829]),
    '392': np.array([-0.00174133, -0.22224286, 0.02866726]),
    '393': np.array([-0.00177275, -0.22274633, 0.02858453]),
    '394': np.array([-0.00178094, -0.22295208, 0.02867156])    
}

class ZJUMocapDataset(Dataset):
    def __init__(self, dataset_root, transform=None, n_input_frames=5, n_test_frames=1, smpl_path=None):
        """
        Args:
            dataset_root (str): Root directory of the dataset
            transform: Optional transform to be applied on images
            n_input_frames (int): Number of frames to sample per scene
        """
        self.dataset_root = dataset_root
        self.transform = transform
        self.n_input_frames = n_input_frames
        self.n_test_frames = n_test_frames
        self.index = self.build_index()
        self.smpl_path = smpl_path

    def build_index(self):
        """
        Build dataset index containing:
            - scene: scene id
            - num_cameras: number of cameras in scene
            - num_frames: total number of frames in scene
        """
        index = []
        for scene in scene_list:
            scene_path = os.path.join(self.dataset_root, scene)
            if not os.path.isdir(scene_path):
                continue

            rgb_bg_dir = os.path.join(scene_path, 'rgb_bg')
            if not os.path.isdir(rgb_bg_dir):
                continue
                
            frame_folders = sorted(os.listdir(rgb_bg_dir))
            if len(frame_folders) == 0:
                continue
                
            first_frame_dir = os.path.join(rgb_bg_dir, frame_folders[0])
            frame_images = [f for f in os.listdir(first_frame_dir) if f.endswith('.jpg') or f.endswith('.png')]
            num_cameras = len(frame_images)
            num_frames = len(frame_folders)
            
            # print(f'Scene {scene}: {num_frames} frames, {num_cameras} cameras per frame')

            index.append({
                'scene': scene,
                'num_cameras': num_cameras,
                'num_frames': num_frames
            })
            
        return index

    def __len__(self):
        return len(self.index)

    def get_frame_data(self, scene, camera_ids, frame_ids):
        """
        获取多个相机和多个帧的数据
        
        参数:
            scene: 场景名称
            camera_ids: 可以是单个相机ID或多个相机ID的列表/数组
            frame_ids: 可以是单个帧ID或多个帧ID的列表/数组
        """
        # 确保camera_ids和frame_ids是可迭代对象
        if not isinstance(camera_ids, (list, np.ndarray)):
            camera_ids = [camera_ids]
        if not isinstance(frame_ids, (list, np.ndarray)):
            frame_ids = [frame_ids]
            
        rgbs = []
        masks = []
        intrinsics = []
        extrinsics = []
        smpls = []
        
        # 遍历所有相机和帧的组合
        for camera_id in camera_ids:
            camera_str = f"{camera_id:03d}"
            for frame_id in frame_ids:
                frame_str = f"{frame_id:06d}"
                
                item = {
                    'scene': scene,
                    'camera': camera_str,
                    'frame': frame_str
                }
                
                rgbs.append(self.load_rgb(item))
                masks.append(self.load_mask(item))
                intrinsics.append(self.load_intrinsic(item))
                extrinsics.append(self.load_extrinsic(item))
                smpls.append(self.load_smpl(item))
        
        # 将所有数据堆叠成批次
        rgbs = np.stack(rgbs)
        masks = np.stack(masks)
        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)
        
        # 合并SMPL参数
        smpl_all = {}
        for smpl in smpls:
            for key in smpl.keys():
                if key not in smpl_all:
                    smpl_all[key] = smpl[key]
                else:
                    smpl_all[key] = np.concatenate([smpl_all[key], smpl[key]], axis=0)
        
        # 处理背景
        masks = masks[..., None] * 255.0
        rgbs = rgbs * masks + (1 - masks) * np.array([255, 255, 255])
        
        # 转换为张量
        rgbs = torch.from_numpy(rgbs.astype(np.float32) / 255.0)
        masks = torch.from_numpy(masks[..., 0].astype(np.float32))
        intrinsics = torch.from_numpy(intrinsics.astype(np.float32))
        extrinsics = torch.from_numpy(extrinsics.astype(np.float32))
        for k, v in smpl_all.items():
            smpl_all[k] = torch.from_numpy(v.astype(np.float32))
              
        return VideoData(
            video=rgbs,
            smpl_parms=smpl_all,
            cam_parms={
                'extrinsic': extrinsics,
                'intrinsic': intrinsics
            },
            width=torch.tensor(rgbs.shape[2], dtype=torch.int32),
            height=torch.tensor(rgbs.shape[3], dtype=torch.int32),
        )

    # def __getitem__(self, idx):
    #     scene_info = self.index[idx]
    #     scene = scene_info['scene']
    #     num_cameras = scene_info['num_cameras']
    #     num_frames = scene_info['num_frames']
        
    #     camera_id = np.random.randint(0, num_cameras)
    #     train_frame_ids = np.random.choice(num_frames, size=self.n_input_frames, replace=False)
    #     test_frame_ids = np.random.choice(num_frames, size=self.n_test_frames, replace=False)

    #     # print(f"scene: {scene}")
    #     # print(f"train_camera_id: {camera_id}, test_camera_id: {camera_id}")
    #     # print(f"train_frame_ids: {train_frame_ids}, test_frame_ids: {test_frame_ids}")
        
    #     train_data = self.get_frame_data(scene, camera_id, train_frame_ids)
    #     test_data = self.get_frame_data(scene, camera_id, test_frame_ids)
        
    #     return {
    #         'train': train_data,
    #         'test': test_data
    #     }

    def __getitem__(self, idx):
        scene_info = self.index[idx]
        scene = scene_info['scene']
        num_cameras = scene_info['num_cameras']
        num_frames = scene_info['num_frames']
        
        frame_id = np.random.randint(0, num_frames)
        
        train_camera_ids = np.random.choice(num_cameras, size=self.n_input_frames, replace=True)
        
        test_camera_ids = np.random.choice(num_cameras, size=self.n_test_frames, replace=True)
        for i in range(self.n_test_frames):
            if test_camera_ids[i] in train_camera_ids:
                test_camera_ids[i] = (test_camera_ids[i] + 1) % num_cameras
        
        # print(f"scene: {scene}")
        # print(f"train_camera_ids: {train_camera_ids}, test_camera_ids: {test_camera_ids}")
        # print(f"frame_id: {frame_id}")
        
        train_data = self.get_frame_data(scene, train_camera_ids, frame_id)
        test_data = self.get_frame_data(scene, test_camera_ids, frame_id)
        
        return {
            'train': train_data,
            'test': test_data
        }

    def load_rgb(self, item):
        scene_path = os.path.join(self.dataset_root, item['scene'])
        rgb_path = os.path.join(scene_path, 'rgb_bg', item['frame'], f"{item['camera']}_{item['frame']}.png")
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def load_mask(self, item):
        scene_path = os.path.join(self.dataset_root, item['scene'])
        mask_path = os.path.join(scene_path, 'mask', item['frame'], f"{item['camera']}_{item['frame']}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        return mask

    def load_intrinsic(self, item):
        scene_path = os.path.join(self.dataset_root, item['scene'])
        intrinsic_path = os.path.join(scene_path, 'intrinsic', f"{item['camera']}.txt")
        intrinsic = np.loadtxt(intrinsic_path)
        return intrinsic

    def load_extrinsic(self, item):
        scene_path = os.path.join(self.dataset_root, item['scene'])
        extrinsic_path = os.path.join(scene_path, 'extrinsic', f"{item['camera']}.txt")
        extrinsic = np.loadtxt(extrinsic_path)
        return extrinsic

    def load_smpl(self, item):
        scene_path = os.path.join(self.dataset_root, item['scene'])
        smpl_path = os.path.join(scene_path, 'smpl_transform', f"{item['frame']}.json")
        with open(smpl_path, 'r') as f:
            smpl = json.load(f)
        
        smpl_np = {}
        for key, value in smpl.items():
            smpl_np[key] = np.array(value)

        smpl_np['body_pose'] = smpl_np['poses']        
        smpl_np['body_pose'][..., :3] = smpl_np['Rh']
        smpl_np['R'] = cv2.Rodrigues(smpl_np['Rh'])[0].astype(np.float32)[None, ...]

        # smpl_model = SMPL(
        #     model_path=self.smpl_path,
        #     batch_size=1
        # )
        
        # beta_tensor = torch.tensor(smpl_np['shapes'], dtype=torch.float32)
        
        # output = smpl_model(
        #     betas=beta_tensor,
        #     body_pose=torch.zeros((1, 69)),     
        #     global_orient=torch.zeros((1, 3)),     
        #     transl=torch.zeros((1, 3))
        # )
        
        # joints = output.joints.detach().numpy()
        
        pelvis_joint = precomputed_pelvis_joints[item['scene']]
        # print(f"pelvis_joint: {pelvis_joint}")


        pelvis_joint_reshaped = pelvis_joint.reshape(3, 1)
        trans_calc = smpl_np['R'][0] @ pelvis_joint_reshaped  # (3,3)@(3,1)->(3,1)
        smpl_np['trans'] = trans_calc.T + smpl_np['Th'] - pelvis_joint  # (1,3)+(1,3)-(1,3)->(1,3)

        smpl_np['beta'] = smpl_np['shapes'].reshape(-1, 10)
        
        # del smpl_np['Rh']
        # del smpl_np['Th']
        return smpl_np


if __name__ == "__main__":
    dataset = ZJUMocapDataset(dataset_root='/home/liubingqi/work/Audio2Avatar/gaussian_avatar/data/zju_mocap')
    print(len(dataset))
    data = dataset[0]
    print(data.keys())
    print("\nVideoData shapes:")
    print("Train data:")
    print(f"video: {data['train'].video.shape}")
    print("smpl_parms:")
    for k, v in data['train'].smpl_parms.items():
        print(f"  {k}: {v.shape}")
    print("cam_parms:")
    for k, v in data['train'].cam_parms.items():
        print(f"  {k}: {v.shape}")
    print(f"width: {data['train'].width}")
    print(f"height: {data['train'].height}")

    print("\nTest data:")
    print(f"video: {data['test'].video.shape}")
    print("smpl_parms:")
    for k, v in data['test'].smpl_parms.items():
        print(f"  {k}: {v.shape}")
    print("cam_parms:")
    for k, v in data['test'].cam_parms.items():
        print(f"  {k}: {v.shape}")
    print(f"width: {data['test'].width}")
    print(f"height: {data['test'].height}")
