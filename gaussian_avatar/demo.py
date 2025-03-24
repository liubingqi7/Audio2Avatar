import os
import lightning as L
from lightning.pytorch import seed_everything
from models.lightning_wrapper import GaussianAvatar
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from datasets.dataset_video import VideoDataset
from datasets.dataset_thuman import BaseDataset
from datasets.dataset_zjumocap import ZJUMocapDataset
from utils.data_utils import collate_fn, collate_fn_thuman, collate_fn_zjumocap
import imageio
import numpy as np
from plyfile import PlyData, PlyElement


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
    parser.add_argument('--dataset', type=str, default='zjumocap', help='Dataset to use for training the gaussian net.')
    parser.add_argument('--n_input_frames', type=int, default=4, help='Number of input frames for training the gaussian net.')
    parser.add_argument('--n_test_frames', type=int, default=4, help='Number of test frames for training the gaussian net.')
    parser.add_argument('--mutiview', action='store_true', help='Whether to use mutiview training.')
    parser.add_argument('--multi_pose', action='store_true', help='Whether to use multi-pose training.')
    args = parser.parse_args()

    return args

def inference():
    args = setup_parser()

    dataset = BaseDataset(
        dataset_root="/home/liubingqi/work/liubingqi/thuman2.0/view5_train",
        scene_list=["/home/liubingqi/work/liubingqi/thuman2.0/train.json"],
        use_smplx=True,
        smpl_dir="/home/liubingqi/work/liubingqi/THuman/THuman2.0_smpl",
        n_input_frames=4,
    )

    dataset = ZJUMocapDataset(
            dataset_root='/home/liubingqi/work/Audio2Avatar/gaussian_avatar/data/zju_mocap',
            smpl_path='/home/liubingqi/work/liubingqi/SMPL_SMPLX/SMPL_models/smpl/SMPL_NEUTRAL.pkl',
            n_input_frames=args.n_input_frames,
            n_test_frames=args.n_test_frames,
            args=args,
        )
    collate_function = collate_fn_zjumocap

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_function,
        shuffle=False,
        num_workers=4
    )

    if args.deform:
        model = GaussianAvatar.load_from_checkpoint(args.ckpt_path, args=args)
    else:
        model = GaussianAvatar(args)
        checkpoint = torch.load(args.ckpt_path)
        gaussian_net_state_dict = {k.replace('gaussian_net.', ''): v for k, v in checkpoint['state_dict'].items() 
                                  if k.startswith('gaussian_net.')}
        model.gaussian_net.load_state_dict(gaussian_net_state_dict)
    model.eval()
    model.to(args.device)

    data = next(iter(dataloader))
    batch, test_batch = data

    with torch.no_grad():
        cano_gaussians = model.gaussian_net(batch, is_train=False)
    
    print(f"cano_gaussians['xyz'].shape: {cano_gaussians['xyz'].shape}")
    # 计算opacity和scale的统计信息并可视化
    opacity = cano_gaussians['opacity'].squeeze(0).detach().cpu().numpy()
    scale = cano_gaussians['scale'].squeeze(0).detach().cpu().numpy()
    
    print("\n高斯点云统计信息:")
    print(f"Opacity 范围: [{opacity.min():.4f}, {opacity.max():.4f}]")
    print(f"Opacity 均值: {opacity.mean():.4f}")
    print(f"Opacity 标准差: {opacity.std():.4f}")
    
    print(f"\nScale 范围: [{scale.min():.4f}, {scale.max():.4f}]")
    print(f"Scale 均值: {scale.mean():.4f}")
    print(f"Scale 标准差: {scale.std():.4f}")

    # 创建直方图可视化
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Opacity直方图
    ax1.hist(opacity, bins=50, alpha=0.75)
    ax1.set_title('Opacity分布')
    ax1.set_xlabel('Opacity值')
    ax1.set_ylabel('频数')
    
    # Scale直方图
    ax2.hist(scale.flatten(), bins=50, alpha=0.75)  
    ax2.set_title('Scale分布')
    ax2.set_xlabel('Scale值')
    ax2.set_ylabel('频数')
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, 'gaussian_stats.png'))
    plt.close()
    
    # 保存cano gaussians到ply文件
    xyz = cano_gaussians['xyz'].squeeze(0).detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = cano_gaussians['color'].squeeze(0).detach().contiguous().cpu().numpy()
    f_dc = (f_dc-0.5)/0.28209479177387814
    opacities = cano_gaussians['opacity'].squeeze(0).detach().cpu().numpy()
    scale = cano_gaussians['scale'].squeeze(0).detach().cpu().numpy() - 3.9
    rotation = cano_gaussians['rot'].squeeze(0).detach().cpu().numpy()

    # 构建属性列表
    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # 添加特征DC和Rest的属性
    for i in range(f_dc.shape[1]):
        attributes.append(f'f_dc_{i}')
    # for i in range(f_rest.shape[1]):
    #     attributes.append(f'f_rest_{i}')
    attributes.extend(['opacity', 'scale_0', 'scale_1', 'scale_2', 
                      'rot_0', 'rot_1', 'rot_2', 'rot_3'])

    dtype_full = [(attribute, 'f4') for attribute in attributes]

    # 合并所有属性
    all_attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, all_attributes))

    # 创建并保存ply文件
    output_path = os.path.join(args.output_dir, 'cano_gaussians.ply')
    os.makedirs(args.output_dir, exist_ok=True)
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    print(f"已保存canonical gaussians到: {output_path}")

    # 读取PLY文件
    plydata = PlyData.read(output_path)

    # 获取点云数据
    vertex = plydata['vertex']
        
    zero_pose = torch.zeros((1, 4, 72)).to(args.device)

    demo_pose = {
        'body_pose': zero_pose,
        'trans': batch.smpl_parms['trans'],
    }

    # 获取当前相机外参
    current_extrinsics = batch.cam_parms['extrinsic']  # [B, N, 4, 4]
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
    batch.cam_parms['extrinsic'] = new_extrinsics

    # 将内参扩展到相同维度
    batch.cam_parms['intrinsic'] = batch.cam_parms['intrinsic'][:, 0:1].repeat(1, num_views, 1, 1)
    
    # 将pose扩展到相同维度
    for k in batch.smpl_parms.keys():
        batch.smpl_parms[k] = batch.smpl_parms[k][:, 0:1].repeat(1, num_views, 1)
    
    # 一次渲染一个视角以最大程度节省显存
    rendered_images = []
    transformed_gaussians = []
    
    for view_idx in range(num_views):
        curr_poses = {k: v[:, view_idx:view_idx+1] for k, v in batch.smpl_parms.items()}
        curr_cam_params = {
            'intrinsic': batch.cam_parms['intrinsic'][:, view_idx:view_idx+1],
            'extrinsic': batch.cam_parms['extrinsic'][:, view_idx:view_idx+1]
        }
        
        curr_rendered, curr_transformed_gaussians = model.animation_net(
            gaussians=cano_gaussians,
            poses=curr_poses,
            cam_params=curr_cam_params,
        )
        
        rendered_images.append(curr_rendered)
        transformed_gaussians.append(curr_transformed_gaussians)

    rendered_image = torch.cat(rendered_images, dim=1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    rendered_image = (rendered_image.detach().cpu().numpy() * 255).astype(np.uint8)
    
    for i in range(rendered_image.shape[1]):
        imageio.imwrite(
            os.path.join(args.output_dir, f'posed_render_{i}.png'),
            rendered_image[0, i]
        )
    # 存储每个视角的transformed gaussians
    for view_idx, transformed_gaussian in enumerate(transformed_gaussians):
        xyz = transformed_gaussian['xyz'].squeeze(0).squeeze(0).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = transformed_gaussian['color'].squeeze(0).squeeze(0).detach().contiguous().cpu().numpy()
        f_dc = (f_dc-0.5)/0.28209479177387814
        opacities = transformed_gaussian['opacity'].squeeze(0).squeeze(0).detach().cpu().numpy()
        scale = transformed_gaussian['scale'].squeeze(0).squeeze(0).detach().cpu().numpy() - 3.9
        rotation = transformed_gaussian['rot'].squeeze(0).squeeze(0).detach().cpu().numpy()

        # 构建属性列表
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # 添加特征DC属性
        for i in range(f_dc.shape[1]):
            attributes.append(f'f_dc_{i}')
        attributes.extend(['opacity', 'scale_0', 'scale_1', 'scale_2', 
                          'rot_0', 'rot_1', 'rot_2', 'rot_3'])

        dtype_full = [(attribute, 'f4') for attribute in attributes]

        # 合并所有属性
        all_attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, all_attributes))

        # 创建并保存ply文件
        output_path = os.path.join(args.output_dir, f'transformed_gaussians_{view_idx}.ply')
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path)
        print(f"已保存transformed gaussians到: {output_path}")

    # store input
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(batch.video.shape[1]):
        imageio.imwrite(
            os.path.join(args.output_dir, f'input_image_{i}.png'),
            (batch.video[0, i].detach().cpu().numpy() * 255.0).astype(np.uint8)
        )

if __name__ == "__main__":
    inference()