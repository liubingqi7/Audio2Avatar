import os
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join
from PIL import Image
import imageio
from utils.data_utils import VideoData
from argparse import ArgumentParser
import sys
import smplx
import trimesh

class VideoDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.clip_length = args.clip_length
        self.clip_overlap = self.clip_length // 2 if args.clip_overlap is None else args.clip_overlap

        self.data_folder = args.data_folder
        self.scene_list = sorted(os.listdir(self.data_folder))
        self.scene_length = len(self.scene_list)
        print("In total {} scenes".format(self.scene_length))

        self.video_data = []
        for scene in self.scene_list:
            scene_path = join(self.data_folder, scene)
            images = sorted(os.listdir(join(scene_path, 'images')))
            camera_path = join(scene_path, 'cam_parms.npz')
            smpl_param_path = join(scene_path, 'smpl_parms.pth')
            mask_path = join(scene_path, 'masks')
            masks = sorted(os.listdir(mask_path))
            num_images = len(images)

            # load camera and smpl params
            camera_params = np.load(camera_path)
            smpl_params = torch.load(smpl_param_path)

            # turn into tensor
            camera_params = {k: torch.from_numpy(np.array(v)) for k, v in camera_params.items()}
            # camera_params = {k: torch.from_numpy(v) for k, v in camera_params.items()}
            camera_params['extrinsic'] = camera_params['extrinsic'].to(torch.float32)
            camera_params['intrinsic'] = camera_params['intrinsic'].to(torch.float32)

            for start in range(0, num_images - self.clip_length + 1, self.clip_length - self.clip_overlap):
                video_chunk = images[start:start + self.clip_length]
                mask_chunk = masks[start:start + self.clip_length]
                camera_chunk = {
                    'extrinsic': camera_params['extrinsic'].unsqueeze(0).repeat(self.clip_length, 1, 1),
                    'intrinsic': camera_params['intrinsic'].unsqueeze(0).repeat(self.clip_length, 1, 1)
                }
                smpl_chunk = {
                    'body_pose': smpl_params['body_pose'][start:start + self.clip_length],
                    'trans': smpl_params['trans'][start:start + self.clip_length],
                    'beta': smpl_params['beta']
                }

                self.video_data.append((scene, video_chunk, camera_chunk, smpl_chunk, mask_chunk))

        print("Loaded {} video chunks from {} scenes.".format(len(self.video_data), self.scene_length))

    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, index):
        return self.getitem_helper(index)
    
    def getitem_helper(self, index):
        scene, video_chunk, camera_chunk, smpl_chunk, mask_chunk = self.video_data[index]
        
        video_path = join(self.data_folder, scene, 'images')
        video_chunk = [join(video_path, img) for img in video_chunk]

        mask_path = join(self.data_folder, scene, 'masks')
        mask_chunk = [join(mask_path, mask) for mask in mask_chunk]
        
        rgbs = []
        for img_path in video_chunk:
            rgbs.append(imageio.v2.imread(img_path))
        
        masks = []
        for i, mask_path in enumerate(mask_chunk):
            mask = imageio.v2.imread(mask_path)
            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            mask = mask[..., np.newaxis]
            if mask.shape[-2] == 3:
                mask = mask[:,:,0] 
            rgbs[i] = (rgbs[i] * mask + (1 - mask) * 255)/255.0

        rgbs = torch.from_numpy(np.array(rgbs, dtype=np.float32)).clamp(0.0, 1.0)

        return VideoData(
            video=rgbs, 
            smpl_parms=smpl_chunk, 
            cam_parms=camera_chunk, 
            width=torch.tensor(rgbs.shape[2]), 
            height=torch.tensor(rgbs.shape[3]),
        )

def parse_args():
    parser = ArgumentParser(description="Video Dataset Parameters")
    
    parser.add_argument('--data_folder', type=str, default="data/gs_data/data/dynvideo_male", 
                        help='Path to the folder containing video data.')
    parser.add_argument('--clip_length', type=int, default=10, 
                        help='Length of each video clip.')
    parser.add_argument('--clip_overlap', type=int, default=5, 
                        help='Overlap between video clips. If None, defaults to half of clip_length.')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    dataset = VideoDataset(args)
    print(len(dataset))
    
    import matplotlib.pyplot as plt

    def visualize_images(video_data):
        video_frames = video_data.video.detach().cpu().numpy()
        num_frames = video_frames.shape[0]

        fig, axes = plt.subplots(2, num_frames, figsize=(15, 5))
        for i in range(num_frames):
            axes[0, i].imshow(video_frames[i])
            axes[0, i].axis('off')

            smpl_data = video_data.smpl_parms
            smpl_data = {k: v.to(torch.device('cuda')) if torch.is_tensor(v) else v for k, v in smpl_data.items()}

            body_pose = smpl_data['body_pose'][i:i+1]
            global_orient = body_pose[:, :3]
            body_pose = body_pose[:, 3:]
            translation = smpl_data['trans'][i:i+1]
            
            # 运行SMPL模型
            smpl_output = smpl_model(
                betas=smpl_data.get('beta', None),
                body_pose=body_pose,
                global_orient=global_orient,
                transl=translation
            )

            vertices = smpl_output.vertices[0].detach().cpu().numpy()
            vertices[:, 1] = -vertices[:, 1]

            smpl_output.vertices = torch.tensor(vertices[None]).to(torch.device('cuda'))
            
            # 获取顶点和面片
            vertices = smpl_output.vertices[0].detach().cpu().numpy()
            faces = smpl_model.faces
            
            # 创建trimesh对象
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # 设置场景和相机
            scene = mesh.scene()
            scene.camera.resolution = (800, 800)
            # 渲染
            rendered_image = scene.save_image(resolution=(800, 800), visible=True)
            rendered_image = Image.open(trimesh.util.wrap_as_stream(rendered_image))

            axes[1, i].imshow(rendered_image)
            axes[1, i].axis('off')

        plt.show()

    smpl_model = smplx.SMPL(
        model_path='/media/qizhu/Expansion/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
        # gender='neutral',
        batch_size=1
    ).to(torch.device('cuda'))


    
    visualize_images(dataset[0])
    print(dataset[0].smpl_parms['body_pose'].shape)