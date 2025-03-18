import os
import lightning as L
from lightning.pytorch import seed_everything
from models.lightning_wrapper import GaussianAvatar
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from datasets.dataset_video import VideoDataset
from datasets.dataset_thuman import BaseDataset
from utils.data_utils import collate_fn
import imageio
import numpy as np

seed_everything(42, workers=True)

def setup_parser():
    parser = ArgumentParser()
    parser = ArgumentParser(description="Video Dataset Parameters")
    
    parser.add_argument('--data_folder', type=str, default="data/gs_data/data/m4c_processed", 
                        help='Path to the folder containing video data.')
    parser.add_argument('--clip_length', type=int, default=1, 
                        help='Length of each video clip.')
    parser.add_argument('--clip_overlap', type=int, default=0, 
                        help='Overlap between video clips. If None, defaults to half of clip_length.')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use for training.')
    parser.add_argument('--smplx_model_path', type=str, default='/media/qizhu/Expansion/SMPL_SMPLX/SMPL_models/smpl/SMPL_NEUTRAL.pkl', 
                        help='Path to the SMPL-X model.')
    parser.add_argument('--image_height', type=int, default=1080,
                        help='')
    parser.add_argument('--image_width', type=int, default=1080, 
                        help='')
    parser.add_argument('--sh_degree', type=int, default=3, 
                        help='')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--rgb', action='store_true', help='Whether to use RGB color')
    parser.add_argument('--use_ckpt', action='store_true', help='Whether to use checkpoint file')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to the checkpoint file.')
    parser.add_argument('--net_ckpt_path', type=str, default=None,
                        help='Path to the Gaussian net checkpoint file.')
    parser.add_argument('--animation_net_ckpt_path', type=str, default=None,
                        help='Path to the animation net checkpoint file.')  
    parser.add_argument('--output_dir', type=str, default='results',
                    help='Output directory for saving rendered images')
    parser.add_argument('--experiment_name', type=str, default='test_lightning',
                    help='Name of the experiment')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--num_iters', type=int, default=2,
                        help='Number of iterations for training the gaussian net.')
    parser.add_argument('--deform', action='store_true', help='Whether to use debug mode')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training the gaussian net.')
    parser.add_argument('--total_steps', type=int, default=200000, help='Total steps for training the gaussian net.')
    
    args = parser.parse_args()

    return args

def inference():
    args = setup_parser()

    dataset = BaseDataset(
        dataset_root="/home/liubingqi/work/liubingqi/thuman2.0/view5_train",
        scene_list=["/home/liubingqi/work/liubingqi/thuman2.0/train.json"],
        use_smplx=True,
        smpl_dir="/home/liubingqi/work/liubingqi/THuman/THuman2.0_smpl",
        n_input_frames=3,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4
    )

    model = GaussianAvatar.load_from_checkpoint(args.ckpt_path, args=args)
    model.eval()
    model.to(args.device)

    data = None
    for i, d in enumerate(dataloader):
        if i == 199:
            data = d
            break

    with torch.no_grad():
        cano_gaussians = model.gaussian_net(data, is_train=False)
    
    print(f"cano_gaussians['xyz'].shape: {cano_gaussians['xyz'].shape}")
    
        
    zero_pose = torch.zeros((1, 4, 72)).to(args.device)

    demo_pose = {
        'body_pose': zero_pose,
        'trans': data.smpl_parms['trans'],
    }

    # 获取当前相机外参
    current_extrinsics = data.cam_parms['extrinsic']  # [B, N, 4, 4]
    B, N = current_extrinsics.shape[:2]
    
    # 定义总视角数
    num_views = 30
    
    # 定义水平旋转和俯仰角度的参数
    angles_y = torch.linspace(0, 2*np.pi, num_views).to(args.device)  # 水平360度旋转
    angles_x = torch.sin(torch.linspace(0, 2*np.pi, num_views)).to(args.device) * 0.5  # 俯仰角度变化(-30度到30度)
    
    # 创建新的外参矩阵列表
    new_extrinsics = []
    
    for b in range(B):
        batch_extrinsics = []
        for view_idx in range(num_views):
            # 获取原始外参
            orig_extrinsic = current_extrinsics[b,0]  # 使用第一个视角作为基准
            
            # 创建绕Y轴的旋转矩阵（水平旋转）
            angle_y = angles_y[view_idx]
            Ry = torch.tensor([[torch.cos(angle_y), 0, torch.sin(angle_y)],
                             [0, 1, 0],
                             [-torch.sin(angle_y), 0, torch.cos(angle_y)]], device=args.device)
            
            # 创建绕X轴的旋转矩阵（俯仰角度）
            angle_x = angles_x[view_idx]
            Rx = torch.tensor([[1, 0, 0],
                             [0, torch.cos(angle_x), -torch.sin(angle_x)],
                             [0, torch.sin(angle_x), torch.cos(angle_x)]], device=args.device)
            
            # 组合新的外参矩阵，先绕Y轴旋转，再绕X轴旋转
            new_extrinsic = torch.eye(4, device=args.device)
            new_extrinsic[:3,:3] = Rx @ Ry @ orig_extrinsic[:3,:3]
            new_extrinsic[:3,3] = orig_extrinsic[:3,3].clone()
            
            batch_extrinsics.append(new_extrinsic)
            
        new_extrinsics.append(torch.stack(batch_extrinsics))
    
    new_extrinsics = torch.stack(new_extrinsics)
    
    # 更新相机参数中的外参
    data.cam_parms['extrinsic'] = new_extrinsics

    # 将内参扩展到相同维度
    data.cam_parms['intrinsic'] = data.cam_parms['intrinsic'][:, 0:1].repeat(1, num_views, 1, 1)

    # 将pose扩展到相同维度
    for k in data.smpl_parms.keys():
        data.smpl_parms[k] = data.smpl_parms[k][:, 0:1].repeat(1, num_views, 1)
    
    rendered_image = model.animation_net(
        gaussians=cano_gaussians,
        poses=data.smpl_parms,
        cam_params=data.cam_parms,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    rendered_image = (rendered_image.detach().cpu().numpy() * 255).astype(np.uint8)
    
    for i in range(rendered_image.shape[1]):
        imageio.imwrite(
            os.path.join(args.output_dir, f'posed_render_{i}.png'),
            rendered_image[0, i]
        )

    print(data.video.shape)

    # store input
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(data.video.shape[1]):
        imageio.imwrite(
            os.path.join(args.output_dir, f'input_image_{i}.png'),
            (data.video[0, i].detach().cpu().numpy() * 255.0).astype(np.uint8)
        )

if __name__ == "__main__":
    inference()