import torch.nn as nn
from .blocks import BasicEncoder, UnetExtractor, GaussianUpdater, GaussianUpdater_2, GaussianDeformer
from smplx import SMPL
import torch
from torch.nn import functional as F
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from models.utils.lbs_utils import batch_rodrigues, batch_rigid_transform, get_edges_from_faces
from utils.graphics_utils import project_gaussians
from models.utils.model_utils import sample_multi_scale_feature
from gaussian_renderer import render_avatars
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import RGB2SH
import numpy as np

class GaussianNet(nn.Module):
    def __init__(self, args):
        super(GaussianNet, self).__init__()
        self.args = args
        self.encoder = UnetExtractor().to(self.args.device)
        if not self.args.rgb:
            self.feat_dim = (self.args.sh_degree + 1) ** 2
        else:
            self.feat_dim = 1
        self.gaussian_updater = GaussianUpdater_2(args, input_dim=288+3+3+4+self.feat_dim*3+3+1, output_color_dim=self.feat_dim*3).to(self.args.device)

        # SMPL model
        self.smpl_model = SMPL(
            model_path=self.args.smplx_model_path,
            # gender=self.args.gender,
            batch_size=1,
        ).to(self.args.device)
        self.lbs_weights = self.smpl_model.lbs_weights # [N, 24]
        self.parents = self.smpl_model.parents # [24]
        self.posedirs = self.smpl_model.posedirs # [24, 3]
        self.J_regressor = self.smpl_model.J_regressor # [24, 6890]
        self.v_template = self.smpl_model.v_template # [6890, 3]
        self.joints = torch.einsum('bik,ji->bjk', [self.v_template.unsqueeze(0), self.J_regressor]) # [1, 24, 3]
        self.edges = torch.tensor(get_edges_from_faces(self.smpl_model.faces)).to(self.args.device) # [2, 20664]
        

        # # 打印显存占用
        # print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        # gaussian init
        self.gaussians = None
        self.num_gaussians = 0  
        

    def forward(self, x):
        rgb_images = x.video.to(self.args.device)

        smpl_params = x.smpl_parms
        cam_params = x.cam_parms

        for k, v in smpl_params.items():
            smpl_params[k] = v.to(self.args.device)
        for k, v in cam_params.items():
            cam_params[k] = v.to(self.args.device)

        self.init_gaussians(smpl_params)

        feats = self.encoder(rgb_images.squeeze(0).permute(0, 3, 1, 2)) # [T, C, H//2, W//2] + [T, C, H//4, W//4] + [T, C, H//8, W//8]

        self.update_gaussians(feats, smpl_params, cam_params, rgb_images, debug=False)

        return self.gaussians

    def init_gaussians(self, smpl_params):
        # Init gaussians on SMPL vertices

        # get smpl vertices
        init_smpl = self.smpl_model(
            betas=smpl_params['beta'].squeeze(0),
            return_verts=True
        )
        verts = init_smpl.vertices
        # self.joints = init_smpl.joints

        # Init color from rgb
        self.num_gaussians = verts.shape[1]

        if not self.args.rgb:
            fused_color = torch.tensor(np.random.random((self.num_gaussians, 3)) / 255.0).float().to(self.args.device)
            features = torch.zeros((fused_color.shape[0], 3, (self.args.sh_degree + 1) ** 2)).float().to(self.args.device)
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            features = torch.zeros((self.num_gaussians, 3)).float().to(self.args.device)

        # Init gaussians on SMPL vertices
        self.gaussians = {
            'xyz': verts.clone().squeeze(0), # [N, 3]
            'rot': torch.zeros(self.num_gaussians, 4).to(self.args.device), # [N, 4]
            'color': features.reshape(self.num_gaussians, -1), # [N, feat_dim*3]
            'scale': torch.log(torch.ones((self.num_gaussians, 3), device="cuda")), # [N, 3]
            'opacity': inverse_sigmoid(0.1 * torch.ones(self.num_gaussians, 1).to(self.args.device)) # [N, 1]
        }
        self.gaussians['rot'][:, 0] = 1

    def update_gaussians(self, feats, smpl_params, cam_params, rgb_images=None, debug=False):
        # Transform, Project, Sample, Update

        # Transform initialized gaussians using LBS
        for i in range(self.args.clip_length):
            transformed_xyz, transformed_rot = self.lbs_transform(smpl_params, index=i)

            # Project transformed gaussians to image plane, [1, N, 2]
            transformed_gaussians = self.gaussians.copy()
            transformed_gaussians['xyz'] = transformed_xyz
            transformed_gaussians['rot'] = transformed_rot
            projected_gaussians = project_gaussians(transformed_gaussians, cam_params['intrinsic'], cam_params['extrinsic'])

            # Sample corresponding features from the feature map
            sampled_features = sample_multi_scale_feature(feats, projected_gaussians['xyz'], index=i)

            if debug:
                draw_gaussians(projected_gaussians, self.args, label='projected')
                # print(f"sampled_features: {sampled_features[...,:3]}")
                # self.gaussians['color'] = sampled_features.squeeze(0)[...,:3]

            # Update gaussians
            self.gaussians = self.gaussian_updater(self.gaussians, sampled_features.squeeze(0)) # , self.edges)
            # print(self.gaussians['color'])

    def lbs_transform(self, smpl_params, index=0):
        body_pose = smpl_params['body_pose'][:, index]
        global_trans = smpl_params['trans'][:, index]

        ident = torch.eye(3, device=self.args.device)
        rot_mats = batch_rodrigues(body_pose.view(-1, 3)).view(
            [1, -1, 3, 3])
        
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([1, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, self.posedirs).view(1, -1, 3)

        J_transformed, A = batch_rigid_transform(rot_mats, self.joints, self.parents)
        W = self.lbs_weights.unsqueeze(dim=0).expand([1, -1, -1])
        num_joints = self.joints.shape[1]
        T = torch.matmul(W, A.view(1, num_joints, 16)) \
            .view(1, -1, 4, 4)
        
        # Add pose offsets to gaussian
        # v_posed = pose_offsets + v_shaped
        gaussians_posed = self.gaussians['xyz'] + pose_offsets # [1, N, 3]

        # Transform gaussian positions and rotations
        homogen_coord = torch.ones([1, gaussians_posed.shape[1], 1], device=self.args.device)
        homogeneous_xyz = torch.cat([
            gaussians_posed,
            homogen_coord
        ], dim=-1)  # [1, N, 4]
        transformed_xyz = torch.matmul(T, homogeneous_xyz.unsqueeze(-1))  # [1, N, 4, 1]
        transformed_xyz = transformed_xyz.squeeze(-1)[..., :3]  # [1, N, 3]
        
        rotation_matrix = T[..., :3, :3]  # [1, N, 3, 3]
        current_quat = self.gaussians['rot']  # [N, 4]
        current_rot_mat = quaternion_to_matrix(current_quat)  # [N, 3, 3]

        new_rot_mat = torch.matmul(rotation_matrix, current_rot_mat.unsqueeze(0))  # [1, N, 3, 3]
        new_quat = matrix_to_quaternion(new_rot_mat.squeeze(0))  # [N, 4]

        # add global translation
        transformed_xyz = transformed_xyz + global_trans
        
        return transformed_xyz, new_quat


class AnimationNet(nn.Module):
    def __init__(self, args):
        super(AnimationNet, self).__init__()
        self.args = args
        self.smpl_model = SMPL(
            model_path=self.args.smplx_model_path,
            # gender=self.args.gender,
            batch_size=1,
        ).to(self.args.device)
        self.lbs_weights = self.smpl_model.lbs_weights # [N, 24]
        self.parents = self.smpl_model.parents # [24]
        self.J_regressor = self.smpl_model.J_regressor # [24, 6890]
        self.v_template = self.smpl_model.v_template # [6890, 3]
        self.joints = torch.einsum('bik,ji->bjk', [self.v_template.unsqueeze(0), self.J_regressor]) # [1, 24, 3]
        self.deformer = GaussianDeformer(self.args)
        self.deforme_none = simple_stack

    def forward(self, gaussians, poses, cam_params):
        body_pose = poses['body_pose']
        # print(f"body_pose.shape: {body_pose.shape}")
        global_trans = poses['trans']
        poses = torch.cat([global_trans, body_pose], dim=-1).squeeze(0)
        # Deform gaussians
        # deformed_gaussians, lbs_offset = self.deformer(gaussians, poses, self.lbs_weights)
        deformed_gaussians, lbs_offset = self.deforme_none(gaussians, poses, self.lbs_weights)

        # Transform gaussians using LBS
        transformed_gaussians = self.lbs_transform(deformed_gaussians, poses, lbs_offset)
        # projected_gaussians = project_gaussians(transformed_gaussians['xyz'], cam_params['intrinsic'], cam_params['extrinsic'])

        # draw_gaussians(projected_gaussians, self.args, label='transformed')

        # render gaussian to image
        rendered_image = []
        for i in range(self.args.clip_length):
            rendered_image.append(render_avatars(transformed_gaussians[i], cam_params, self.args, debug=False))

        rendered_image = torch.stack(rendered_image)
        return rendered_image
    
    def lbs_transform(self, gaussians, poses, lbs_offset):
        # print(f"poses.shape: {poses.shape}")
        global_trans = poses[:, :3]
        body_pose = poses[:, 3:]
        B, _ = body_pose.shape
        ident = torch.eye(3, device=self.args.device)
        rot_mats = batch_rodrigues(body_pose.reshape(-1, 3)).view(
            [B, -1, 3, 3])
        
        # pose_feature = (rot_mats[:, 1:, :, :] - ident).view([B, -1])
        # # (N x P) x (P, V * 3) -> N x V x 3
        # pose_offsets = torch.matmul(
        #     pose_feature, self.posedirs).view(1, -1, 3)
        # print(f"rot_mats: {rot_mats.shape}")
        # print(f" global_trans.shape: { global_trans.shape}")

        J_transformed, A = batch_rigid_transform(rot_mats, self.joints.repeat(B, 1, 1), self.parents)

        # print(f"lbs_offset.shape: {lbs_offset}")
        W = self.lbs_weights.unsqueeze(dim=0).expand([B, -1, -1])#  + lbs_offset
        num_joints = self.joints.shape[1]
        T = torch.matmul(W, A.view(B, num_joints, 16)) \
            .view(B, -1, 4, 4)
        
        # Add pose offsets to gaussian
        # v_posed = pose_offsets + v_shaped
        # print(f"gaussians['xyz'].shape: {gaussians['xyz'].shape}")
        gaussians_posed = gaussians['xyz']

        # Transform gaussian positions and rotations
        homogen_coord = torch.ones([B, gaussians_posed.shape[1], 1], device=self.args.device)
        homogeneous_xyz = torch.cat([
            gaussians_posed,
            homogen_coord
        ], dim=-1)  # [B, N, 4]
        transformed_xyz = torch.matmul(T, homogeneous_xyz.unsqueeze(-1))  # [B, N, 4, 1]
        transformed_xyz = transformed_xyz.squeeze(-1)[..., :3]  # [B, N, 3]
        
        rotation_matrix = T[..., :3, :3]  # [B, N, 3, 3]
        current_quat = gaussians['rot']  # [B, N, 4]
        current_rot_mat = quaternion_to_matrix(current_quat)  # [B, N, 3, 3]

        new_rot_mat = torch.matmul(rotation_matrix, current_rot_mat.unsqueeze(0))  # [B, N, 3, 3]
        new_quat = matrix_to_quaternion(new_rot_mat.squeeze(0))  # [B, N, 4]

        # # add global translation
        # print(f"global_trans.shape: {global_trans.shape}")
        # print(f"transformed_xyz.shape: {transformed_xyz.shape}")
        transformed_xyz = transformed_xyz + global_trans.unsqueeze(1)

        transformed_gaussians = []
        for i in range(B):
            transformed_gaussians.append({
                'xyz': transformed_xyz[i],
                'rot': new_quat[i],
                'color': gaussians['color'][i],
                'scale': gaussians['scale'][i],
                'opacity': gaussians['opacity'][i]
            })
        # transformed_gaussians = {
        #     'xyz': transformed_xyz,  # [B, N, 3]
        #     'rot': new_quat,  # [B, N, 4] 
        #     'color': gaussians['color'],  # [B, N, 3]
        #     'scale': gaussians['scale'],  # [B, N, 3]
        #     'opacity': gaussians['opacity']  # [B, N, 1]
        # }
        
        return transformed_gaussians


def simple_stack(gaussians, poses, lbs_weights):
    B = poses.shape[0]
    N = gaussians['xyz'].shape[0]

    deformed_gaussians = []
    lbs_offsets = torch.zeros([B, N, 24], device=lbs_weights.device)

    for i in range(B):
        deformed_gs = {}
        deformed_gs['xyz'] = gaussians['xyz']
        deformed_gs['scale'] = gaussians['scale']
        deformed_gs['rot'] = gaussians['rot']
        deformed_gs['opacity'] = gaussians['opacity']
        deformed_gs['color'] = gaussians['color']

        deformed_gaussians.append(deformed_gs)

    combined_deformed_gs = {}
    for key in deformed_gaussians[0].keys():
        combined_deformed_gs[key] = torch.stack([d[key] for d in deformed_gaussians], dim=0)

    return combined_deformed_gs, lbs_offsets


def draw_gaussians(projected_gaussians, args, label='projected'):
    # 创建一个空白图像
    img = torch.zeros((1080, 1080, 3), device=args.device)
    
    # 将投影点的坐标四舍五入到最近的整数
    projected_points = projected_gaussians['xyz'].round().long()
    
    # 确保点在图像范围内
    mask = (projected_points[..., 0] >= 0) & (projected_points[..., 0] < 1080) & \
            (projected_points[..., 1] >= 0) & (projected_points[..., 1] < 1080)
    
    # 只保留在图像范围内的点
    valid_points = projected_points[mask]
    
    # 在这些点的位置画白色点
    if valid_points.shape[0] > 0:
        img[valid_points[:, 1], valid_points[:, 0]] = torch.tensor([1.0, 1.0, 1.0], device=args.device)
    
    # 保存图像
    import torchvision
    save_path = f"test_images/{label}.png"
    torchvision.utils.save_image(img.permute(2, 0, 1), save_path)

    # save original image
    # save_path = f"test_images/orig.png"
    # torchvision.utils.save_image(rgb_images[0, i].permute(2, 0, 1), save_path)
