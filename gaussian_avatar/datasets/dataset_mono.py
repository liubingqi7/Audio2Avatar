import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from os.path import join
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from argparse import ArgumentParser, Namespace
from arguments import DataParams
from scene.cameras import Camera


class MonoDataset_train(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_parms,
                 device = torch.device('cuda:0')):
        super(MonoDataset_train, self).__init__()

        ####
        print("Source path:", dataset_parms.source_path)
        print("Data folder path:", join(dataset_parms.source_path, 'train'))
        print("SMPL file path:", join(dataset_parms.source_path, 'train', 'smpl_parms.pth'))

        self.dataset_parms = dataset_parms

        self.data_folder = join(dataset_parms.source_path, 'train')
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

        print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
        self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))

        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_parms.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]

            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
            if not torch.is_tensor(self.smpl_data['body_pose']):
                self.pose_data = torch.from_numpy(self.pose_data)
            if not torch.is_tensor(self.smpl_data['trans']):
                self.transl_data = torch.from_numpy(self.transl_data)
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
            if not torch.is_tensor(self.smpl_data['body_pose']):
                self.pose_data = torch.from_numpy(self.pose_data)
            if not torch.is_tensor(self.smpl_data['trans']):
                self.transl_data = torch.from_numpy(self.transl_data)


        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):
        pose_idx, name_idx = self.name_list[index]

        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)
        
        cam_path = join(self.data_folder, 'cam_parms', name_idx + '.npz')

        if not self.dataset_parms.no_mask:
            mask_path = join(self.data_folder, 'masks', name_idx + '.' + self.mask_fix)

        if not self.dataset_parms.cam_static:
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extr_npy[:3, 3], np.float32)
            intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        else:
            R = self.R
            T = self.T
            intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        image = Image.open(image_path)
        width, height = image.size

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        if not self.dataset_parms.no_mask:
            mask = np.array(Image.open(mask_path))

            if len(mask.shape) <3:
                mask = mask[...,None]

            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            color_img = image * mask + (1 - mask) * 255
            image = Image.fromarray(np.array(color_img, dtype=np.byte), "RGB")

        
        data_item = dict()

        resized_image = torch.from_numpy(np.array(image)) / 255.0
        if len(resized_image.shape) == 3:
            resized_image =  resized_image.permute(2, 0, 1)
        else:
            resized_image =  resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

        original_image = resized_image.clamp(0.0, 1.0)

        data_item['original_image'] = original_image
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        data_item['timestep'] = pose_idx

        camera_info = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            bg=None,
            image_width=width,
            image=original_image,
            image_height=height,
            image_path=image_path,
            image_name=name_idx,
            uid=pose_idx,
            trans=self.trans,
            scale=self.scale,
            timestep=pose_idx,
            world_view_transform=world_view_transform,
            projection_matrix=projection_matrix,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center
        )
        
        return camera_info

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = DataParams(parser)
    dataset = MonoDataset_train(lp)
    print(dataset[0])
    
