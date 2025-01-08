import os
import torch
import smplx
import numpy as np
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from scene.dataset_readers import readCameraFromMonoDataset, readSMPLMeshes

def load_smpl_params(data_path):
    """加载SMPL参数"""
    smpl_path = os.path.join(data_path, 'smpl_parms.pth')
    print(f"从{smpl_path}加载SMPL数据")
    smpl_data = torch.load(smpl_path)

    return smpl_data

def verify_smpl_params_from_pkl(data_path):
    smpl_data = load_smpl_params_from_pkl(data_path)
    verify_smpl_params(smpl_data)

def load_smpl_params_from_pkl(smpl_path):
    """加载SMPL参数"""
    print(f"从{smpl_path}加载SMPL数据")
    smpl_data = pickle.load(open(smpl_path, 'rb'))

    return smpl_data

def verify_smpl_params(smpl_data, timestep=None):
    device = torch.device('cuda')

    smpl_model = smplx.SMPL(
        model_path='/media/qizhu/Expansion/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
        batch_size=1
    ).to(device)

    # 将smpl data转换为tensor并确保维度正确
    for k, v in smpl_data.items():
        if isinstance(v, np.ndarray):
            smpl_data[k] = torch.from_numpy(v).float()
        elif isinstance(v, list):
            smpl_data[k] = torch.tensor(v).float()
    
    # 确保所有输入tensor都有正确的batch维度
    if len(smpl_data['betas'].shape) == 1:
        smpl_data['betas'] = smpl_data['betas'].unsqueeze(0)
    if len(smpl_data['body_pose'].shape) == 1:
        smpl_data['body_pose'] = smpl_data['body_pose'].unsqueeze(0)
    if len(smpl_data['global_orient'].shape) == 1:
        smpl_data['global_orient'] = smpl_data['global_orient'].unsqueeze(0)
    if len(smpl_data['translation'].shape) == 1:
        smpl_data['translation'] = smpl_data['translation'].unsqueeze(0)

    # 移动到GPU
    smpl_data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in smpl_data.items()}

    # 打印形状以进行调试
    print("Input shapes:")
    print(f"betas: {smpl_data['betas'].shape}")
    print(f"body_pose: {smpl_data['body_pose'].shape}")
    print(f"global_orient: {smpl_data['global_orient'].shape}")
    print(f"translation: {smpl_data['translation'].shape}")

    smpl_output = smpl_model(
        betas=smpl_data['betas'],
        body_pose=smpl_data['body_pose'],
        global_orient=smpl_data['global_orient'],
        transl=smpl_data['translation']
    )

    # 翻转y轴
    vertices = smpl_output.vertices[0].detach().cpu().numpy()
    vertices[:, 1] = -vertices[:, 1]

    smpl_output.vertices = torch.tensor(vertices[None]).to(device)
    
    # 获取顶点和面片
    vertices = smpl_output.vertices[0].detach().cpu().numpy()
    faces = smpl_model.faces
    
    # 创建trimesh对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 设置场景和相机
    scene = mesh.scene()
    
    # 渲染
    rendered_image = scene.save_image(resolution=(800, 800), visible=True)
    rendered_image = Image.open(trimesh.util.wrap_as_stream(rendered_image))

    # save the rendered image
    if timestep is not None:
        rendered_image.save(f"smpl_verification_{timestep}.png")
    else:
        rendered_image.save(f"smpl_verification_{smpl_data['timestep']}.png")

    return 

def main():
    # 设置路径
    data_path = 'data/gs_data/data/dynvideo_male/train'
    device = torch.device('cuda')
    
    # 加载SMPL模型
    smpl_model = smplx.SMPL(
        model_path='/media/qizhu/Expansion/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
        # gender='neutral',
        batch_size=1
    ).to(device)
    
    # 加载SMPL参数
    smpl_data = load_smpl_params(data_path)
    smpl_data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in smpl_data.items()}

    # 打印SMPL参数信息
    print("\nSMPL参数信息:")
    for key, value in smpl_data.items():
        if torch.is_tensor(value):
            print(f"{key}: shape = {value.shape}")
        else:
            print(f"{key}: {type(value)}")

        # 打印第一帧数据示例
        if len(value) > 0:
            print(f"第一帧数据示例:\n{value[0]}\n")
    
    # 加载相机参数
    camera_path = os.path.join(data_path, 'cam_parms.npz')
    print(f"\n从{camera_path}加载相机参数")
    camera_data = np.load(camera_path)
    
    # 打印相机参数信息
    print("\n相机参数信息:")
    for key, value in camera_data.items():
        print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")
        if len(value) > 0:
            print(f"第一帧数据示例:\n{value}\n")

    # 读取图像文件
    image_dir = os.path.join(data_path, 'images')
    image_files = sorted(os.listdir(image_dir))
    
    # 从不同起始位置取样10次,每次取5张连续图片
    start_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 
                     50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 
                     100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 
                     150, 155, 160, 165, 170, 175, 180, 185, 190, 195]  # 不同的起始位置
    
    for start_idx in start_indices:
        plt.figure(figsize=(15, 5))
        selected_files = image_files[start_idx:start_idx+5]  # 从start_idx开始连续取5张
        
        for idx, img_file in enumerate(selected_files):
            # 读取原始图像
            img_path = os.path.join(image_dir, img_file)
            orig_img = Image.open(img_path)
            
            # 获取对应帧的SMPL参数
            frame_idx = start_idx + idx
            body_pose = smpl_data['body_pose'][frame_idx:frame_idx+1]
            global_orient = body_pose[:, :3]
            body_pose = body_pose[:, 3:]
            translation = smpl_data['trans'][frame_idx:frame_idx+1]
            
            # 运行SMPL模型
            smpl_output = smpl_model(
                betas=smpl_data.get('beta', None),
                body_pose=body_pose,
                global_orient=global_orient,
                transl=translation
            )

            # 翻转y轴
            vertices = smpl_output.vertices[0].detach().cpu().numpy()
            vertices[:, 1] = -vertices[:, 1]
            
            #输出vertice的范围
            print(f"vertices range:")
            print(f"X: [{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
            print(f"Y: [{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
            print(f"Z: [{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

            smpl_output.vertices = torch.tensor(vertices[None]).to(device)
            
            # 获取顶点和面片
            vertices = smpl_output.vertices[0].detach().cpu().numpy()
            faces = smpl_model.faces
            
            # 创建trimesh对象
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # 设置场景和相机
            scene = mesh.scene()
            
            # 渲染
            rendered_image = scene.save_image(resolution=(800, 800), visible=True)
            rendered_image = Image.open(trimesh.util.wrap_as_stream(rendered_image))
            
            # 显示结果
            plt.subplot(2, 5, idx+1)
            plt.imshow(orig_img)
            plt.title(f'Original {frame_idx}')
            plt.axis('off')
            
            plt.subplot(2, 5, idx+6)
            plt.imshow(rendered_image)
            plt.title(f'Rendered {frame_idx}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'./smpl_verification/smpl_verification_{start_idx}.png')
        plt.close()
        
        print(f"验证结果已保存到 smpl_verification_{start_idx}.png")

    visualize_point_cloud(camera_data['intrinsic'], camera_data['extrinsic'])

# 读取和可视化点云
def visualize_point_cloud(cam_intrinsics, cam_extrinsics):
    # Load point cloud file
    point_cloud = trimesh.load('output_gaussians/point_cloud/point_cloud.ply')
    print(point_cloud)
    points = point_cloud.vertices
    
    print(f"Point Cloud Info:")
    print(f"Number of points: {len(points)}")
    print(f"Point cloud data type: {points.dtype}")
    print(f"Point cloud data range:")
    print(f"X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
    print(f"Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]") 
    print(f"Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

    # Create new figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot original point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(points[:,0], points[:,1], points[:,2], 
                        c=points[:,2],
                        cmap='viridis',
                        s=1)
    ax1.set_title('Original Point Cloud')
    
    # Draw camera coordinate frame
    cam_scale = 0.3 * np.max(np.abs(points))
    cam_pos = np.zeros(3)
    # Draw coordinate axes
    ax1.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
              cam_scale, 0, 0, color='r', alpha=0.5)
    ax1.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
              0, cam_scale, 0, color='g', alpha=0.5)
    ax1.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
              0, 0, cam_scale, color='b', alpha=0.5)
    
    # Transform points to camera coordinate system
    points_homo = np.hstack([points, np.ones((len(points), 1))])  # Homogeneous coordinates
    cam_points = (cam_extrinsics @ points_homo.T).T[:, :3]  # Transform to camera coordinates
    
    # Plot point cloud in camera coordinates
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(cam_points[:,0], cam_points[:,1], cam_points[:,2], 
                        c=cam_points[:,2],
                        cmap='viridis',
                        s=1)
    ax2.set_title('Point Cloud in Camera Space')
    
    # Draw camera coordinate frame
    ax2.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
              cam_scale, 0, 0, color='r', alpha=0.5)
    ax2.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
              0, cam_scale, 0, color='g', alpha=0.5)
    ax2.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
              0, 0, cam_scale, color='b', alpha=0.5)
    
    # Project to image plane
    fx, fy = cam_intrinsics[0,0], cam_intrinsics[1,1]
    cx, cy = cam_intrinsics[0,2], cam_intrinsics[1,2]
    
    # Perspective projection
    img_points = np.zeros((len(cam_points), 2))
    valid_mask = cam_points[:,2] > 0  # Keep only points with z>0
    img_points[valid_mask,0] = fx * cam_points[valid_mask,0] / cam_points[valid_mask,2] + cx
    img_points[valid_mask,1] = fy * cam_points[valid_mask,1] / cam_points[valid_mask,2] + cy
    
    # Plot projection result
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(img_points[valid_mask,0], img_points[valid_mask,1],
                        c=cam_points[valid_mask,2],
                        cmap='viridis',
                        s=1)
    ax3.set_xlim(0, 800)  # Assume 800x800 image
    ax3.set_ylim(0, 800)
    ax3.invert_yaxis()  # Image coordinate system y-axis points down
    ax3.set_title('Projected to Image Plane')
    
    plt.tight_layout()
    plt.savefig('point_cloud_projection.png')
    plt.close()
    print(f"点云投影结果已保存到 point_cloud_projection.png")

def verify_dataset_loading(data_path, split='train'):
    """验证数据集加载的一致性"""
    print(f"\n=== 验证数据集加载 ({split}) ===")
    
    # 1. 验证图像加载顺序
    image_dir = os.path.join(data_path, split, 'images')
    image_files = sorted(os.listdir(image_dir))
    print(f"\n图像文件顺序 (前5个):")
    for i, img in enumerate(image_files[:5]):
        print(f"Frame {i}: {img}")
    
    # 2. 验证SMPL参数
    smpl_path = os.path.join(data_path, split, 'smpl_parms.pth')
    smpl_data = torch.load(smpl_path)
    print(f"\nSMPL参数形状:")
    for k, v in smpl_data.items():
        if torch.is_tensor(v):
            print(f"{k}: {v.shape}")
    
    # 3. 加载相机参数
    cam_path = os.path.join(data_path, split, 'cam_parms.npz')
    cam_data = np.load(cam_path)
    print(f"\n相机参数形状:")
    for k, v in cam_data.items():
        print(f"{k}: {v.shape}")
    
    # 4. 验证数据对应关系
    cam_infos = readCameraFromMonoDataset(data_path, split)
    mesh_infos = readSMPLMeshes(data_path, split)
    
    print(f"\n数据对应关系验证 (前5帧):")
    for i in range(5):
        print(f"\nFrame {i}:")
        print(f"Camera info - timestep: {cam_infos[i].timestep}, image: {cam_infos[i].image_name}")
        if i in mesh_infos:
            print(f"Mesh info available - translation: {mesh_infos[i]['translation'][:3]}")
            print(f"           body_pose shape: {mesh_infos[i]['body_pose'].shape}")
        else:
            print("No mesh info found!")
            
    return cam_infos, mesh_infos

def verify_smpl_rendering(cam_infos, mesh_infos, save_dir="verification_results"):
    """验证SMPL渲染结果与原始图像的对应关系"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(5, len(cam_infos))):
        cam_info = cam_infos[i]
        mesh_info = mesh_infos[cam_info.timestep]
        
        # 保存原始图像
        orig_img = cam_info.image
        orig_img.save(os.path.join(save_dir, f"frame_{i}_original.png"))
        
        # 渲染SMPL
        verify_smpl_params(mesh_info, timestep=cam_info.timestep)
        
        print(f"Frame {i} verification saved to {save_dir}")

if __name__ == "__main__":
    data_path = "/home/qizhu/Desktop/Work/MotionGeneration/Audio2Avatar/gaussian_avatar/data/gs_data/data/dynvideo_male"
    cam_infos, mesh_infos = verify_dataset_loading(data_path)
    verify_smpl_rendering(cam_infos, mesh_infos)
