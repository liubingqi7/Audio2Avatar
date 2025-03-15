import torch.nn as nn
from .blocks import BasicEncoder, UnetExtractor, GaussianUpdater, GaussianUpdater_2, GaussianDeformer, EncoderLayer
from smplx import SMPL
import torch
from torch.nn import functional as F
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from models.utils.lbs_utils import batch_rodrigues, batch_rigid_transform, get_edges_from_faces
from utils.graphics_utils import project_gaussians, project_xyz
from models.utils.model_utils import sample_multi_scale_feature
from gaussian_renderer import render_batch
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import RGB2SH
import numpy as np
from models.utils.subdivide_smpl import subdivide_smpl_model
import nvdiffrast.torch
import cv2
import copy


from utils.graphics_utils import clip_T_world


class GaussianNet(nn.Module):
    def __init__(self, args):
        super(GaussianNet, self).__init__()
        self.args = args
        self.encoder = UnetExtractor().to(self.args.device)
        if not self.args.rgb:
            self.color_dim = (self.args.sh_degree + 1) ** 2
        else:
            self.color_dim = 1
        self.gaussian_updater = GaussianUpdater_2(args, input_dim=(288+3)*self.args.clip_length+3+4+self.color_dim*3+3+1, output_color_dim=self.color_dim*3).to(self.args.device)

        self.cross_view_attn = EncoderLayer(291, 3, 291, 291)

        # SMPL model
        self.smpl_model = SMPL(
            model_path=self.args.smplx_model_path,
            # gender=self.args.gender,
            batch_size=1,
            create_global_orient=False,
            create_betas=False,
            create_body_pose=False,
            create_transl=False,
        ).to(self.args.device)

        # subdivide smpl model
        self.smpl_model = subdivide_smpl_model(self.smpl_model, n_iter=1, SMPL_PATH=self.args.smplx_model_path)

        self.lbs_weights = self.smpl_model.lbs_weights # [N, 24]
        self.parents = self.smpl_model.parents # [24]
        self.posedirs = self.smpl_model.posedirs # [24, 3]
        self.J_regressor = self.smpl_model.J_regressor # [24, 6890]
        self.v_template = self.smpl_model.v_template # [6890, 3]
        self.shapedirs = self.smpl_model.shapedirs
        self.joints = torch.einsum('bik,ji->bjk', [self.v_template.unsqueeze(0), self.J_regressor]) # [1, 24, 3]
        # self.edges = torch.tensor(get_edges_from_faces(self.smpl_model.faces)).to(self.args.device) # [2, 20664]
        self.faces = self.smpl_model.faces # [20664, 3]
        
        self.gaussians = None
        self.num_gaussians = 0  

        self.rasterize_context = nvdiffrast.torch.RasterizeCudaContext(device='cuda')
        self.num_iters = args.num_iters

    def forward(self, x, is_train=True):
        rgb_images = x.video.to(self.args.device)

        smpl_params = x.smpl_parms
        cam_params = x.cam_parms

        for k, v in smpl_params.items():
            smpl_params[k] = v.to(self.args.device)
        for k, v in cam_params.items():
            cam_params[k] = v.to(self.args.device)

        B, T, H, W, _ = rgb_images.shape

        feats = self.encoder(rgb_images.reshape(B*T, H, W, 3).permute(0, 3, 1, 2)) # [B*T, 3, H, W] + [B*T, C, H//2, W//2] + [B*T, C, H//4, W//4] + [B*T, C, H//8, W//8]

        curr_gaussians = self.init_gaussians(smpl_params)

        if is_train:
            all_gaussians = []

        for i in range(self.args.num_iters):
            curr_gaussians = self.update_gaussians(curr_gaussians, feats, smpl_params, cam_params, rgb_images, debug=False)

            # save gaussians while training
            tmp_gs = {}
            for key, value in curr_gaussians.items():
                if isinstance(value, torch.Tensor):
                    tmp_gs[key] = value.clone()
                else:
                    tmp_gs[key] = value 

            all_gaussians.append(tmp_gs)

        if is_train:
            return curr_gaussians, all_gaussians
        else:
            return curr_gaussians

    def init_gaussians(self, smpl_params):
        B, N_pose, _ = smpl_params['body_pose'].shape
        blend_shape = torch.einsum('bl,mkl->bmk', [smpl_params['beta'][:, 0], self.shapedirs])
        v_shaped = self.v_template + blend_shape
        
        # regress J from v_shaped
        joints = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor]) # [B, 24, 3]
        
        # Init color from rgb
        self.num_gaussians = v_shaped.shape[1]

        if not self.args.rgb:
            fused_color = torch.tensor(np.random.random((B, self.num_gaussians, 3)) / 255.0).float().to(self.args.device)
            features = torch.zeros((B, fused_color.shape[1], 3, (self.args.sh_degree + 1) ** 2)).float().to(self.args.device)
            features[:, :, :3, 0] = fused_color
            features[:, :, 3:, 1:] = 0.0
        else:
            features = torch.zeros((B, self.num_gaussians, 3)).float().to(self.args.device)

        # Init gaussians on SMPL vertices
        init_gaussians = {
            'xyz': v_shaped.clone(), # [B, N_gs, 3]
            'rot': torch.zeros(B, self.num_gaussians, 4).to(self.args.device), # [B, N_gs, 4]
            'color': features.reshape(B, self.num_gaussians, -1), # [B, N_gs, color_dim*3]
            'scale': torch.log(torch.ones((B, self.num_gaussians, 3), device="cuda")), # [B, N_gs, 3]
            'opacity': inverse_sigmoid(0.1 * torch.ones(B, self.num_gaussians, 1).to(self.args.device)), # [B, N_gs, 1]
            'feats': torch.zeros((B, self.num_gaussians, 32), device="cuda"), # [B, N_gs, feat_dim]
            'lbs_weights': self.lbs_weights.clone().unsqueeze(0).expand(B, -1, -1) # [B, N_gs, 24]
        }
        init_gaussians['rot'][..., 0] = 1

        return init_gaussians

    def update_gaussians(self, gaussians, feats, smpl_params, cam_params, rgb_images=None, debug=False):
        # Transform, Project, Sample, Update
        B, T, H, W, _ = rgb_images.shape
        B, N_pose, _ = smpl_params['body_pose'].shape
        # Transform initialized gaussians using LBS
        transformed_xyz, transformed_rot = self.lbs_transform(gaussians, smpl_params)

        # project transformed gaussians to image plane
        projected_gaussians_uv = project_xyz(transformed_xyz.reshape(-1, self.num_gaussians, 3), cam_params['intrinsic'].reshape(-1, 3, 3), cam_params['extrinsic'].reshape(-1, 4, 4)).reshape(B, N_pose, -1, 2)
        # print(f"projected_gaussians_uv shape: {projected_gaussians_uv.shape}")

        if debug:
            # draw projected gaussians
            draw_gaussians(projected_gaussians_uv, rgb_images)

        # sample features according to projected_gaussians_uv
        projected_gaussians_uv[..., 0] = projected_gaussians_uv[..., 0]/ W * 2 - 1.0
        projected_gaussians_uv[..., 1] = projected_gaussians_uv[..., 1]/ H * 2 - 1.0
        sampled_features = sample_multi_scale_feature(feats, projected_gaussians_uv.reshape(B*T, self.num_gaussians, 2)).reshape(B, T, self.num_gaussians, -1)
        # print(f"sampled_features shape: {sampled_features.shape}")

        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(B*self.num_gaussians, T, -1)

        # if self.args.visibility_check:
        #     visibility_map = self.visibility_check(transformed_gaussians, cam_params['intrinsic'][:, i, :, :], cam_params['extrinsic'][:, i, :, :])
            
        #     # soft mask from visibility map
        #     epsilon = 1e-3
        #     soft_mask = visibility_map.float() * 1.0 + (1 - visibility_map.float()) * epsilon
        #     sampled_features = sampled_features * soft_mask.unsqueeze(-1)

        # cross-view attention
        sampled_features = self.cross_view_attn(sampled_features, sampled_features, sampled_features)
        sampled_features = sampled_features.reshape(B, self.num_gaussians, T, -1).reshape(B, self.num_gaussians, -1)

        updated_gaussians = self.gaussian_updater(gaussians, sampled_features)

        return updated_gaussians

    def lbs_transform(self, gaussians, smpl_params):
        B, N_pose, _ = smpl_params['body_pose'].shape
        body_pose = smpl_params['body_pose']
        global_trans = smpl_params['trans']

        # print(f"body_pose shape: {body_pose.shape}")
        # print(f"global_trans shape: {global_trans.shape}")

        ident = torch.eye(3, device=self.args.device)
        rot_mats = batch_rodrigues(body_pose.reshape(-1, 3)).view(
            [B, N_pose, -1, 3, 3])
        
        # print(f"rot_mats shape: {rot_mats.shape}")
        
        pose_feature = (rot_mats[:, :, 1:, :, :] - ident).view([B, N_pose, -1])
        # (B x N_pose x P) x (P, V * 3) -> B x N_pose x V x 3
        pose_offsets = torch.matmul(
            pose_feature, self.posedirs).view(B, N_pose, -1, 3)

        # print(f"pose_feature shape: {pose_feature.shape}")
        # print(f"pose_offsets shape: {pose_offsets.shape}")

        J_transformed, A = batch_rigid_transform(rot_mats.view(B*N_pose, -1, 3, 3), 
                                               self.joints.repeat(B*N_pose, 1, 1), 
                                               self.parents)
        A = A.view(B, N_pose, -1, 4, 4)
        
        # print(f"J_transformed shape: {J_transformed.shape}")
        # print(f"A shape: {A.shape}")
        
        W = self.lbs_weights.unsqueeze(0).unsqueeze(0).expand([B, N_pose, -1, -1])
        num_joints = self.joints.shape[1]
        T = torch.matmul(W, A.view(B, N_pose, num_joints, 16)) \
            .view(B, N_pose, -1, 4, 4)
        
        # print(f"W shape: {W.shape}")
        # print(f"T shape: {T.shape}")

        # Add pose offsets to gaussian
        gaussians_posed = gaussians['xyz'].unsqueeze(1).expand(-1, N_pose, -1, -1) + pose_offsets

        # print(f"gaussians_posed shape: {gaussians_posed.shape}")

        # Transform gaussian positions and rotations
        homogen_coord = torch.ones([B, N_pose, gaussians_posed.shape[2], 1], device=self.args.device)
        homogeneous_xyz = torch.cat([
            gaussians_posed,
            homogen_coord
        ], dim=-1)  # [B, N_pose, N, 4]
        transformed_xyz = torch.matmul(T, homogeneous_xyz.unsqueeze(-1))  # [B, N_pose, N, 4, 1]
        transformed_xyz = transformed_xyz.squeeze(-1)[..., :3]  # [B, N_pose, N, 3]
        
        # print(f"homogeneous_xyz shape: {homogeneous_xyz.shape}")
        # print(f"transformed_xyz shape: {transformed_xyz.shape}")
        
        rotation_matrix = T[..., :3, :3]  # [B, N_pose, N, 3, 3]
        current_quat = gaussians['rot'].unsqueeze(1).expand(-1, N_pose, -1, -1)  # [B, N_pose, N, 4]
        current_rot_mat = quaternion_to_matrix(current_quat)  # [B, N_pose, N, 3, 3]

        # print(f"rotation_matrix shape: {rotation_matrix.shape}")
        # print(f"current_quat shape: {current_quat.shape}")
        # print(f"current_rot_mat shape: {current_rot_mat.shape}")

        new_rot_mat = torch.matmul(rotation_matrix, current_rot_mat)  # [B, N_pose, N, 3, 3]
        new_quat = matrix_to_quaternion(new_rot_mat)  # [B, N_pose, N, 4]

        # print(f"new_rot_mat shape: {new_rot_mat.shape}")
        # print(f"new_quat shape: {new_quat.shape}")

        # add global translation
        transformed_xyz = transformed_xyz + global_trans.unsqueeze(2)
        
        return transformed_xyz, new_quat
    
    def visibility_check(self, gaussians, K, E):
        xyz = gaussians['xyz']
        resolution = [self.args.image_height, self.args.image_width]
        visibility_map = xyz.new_zeros(xyz.shape[0], xyz.shape[1])

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            xyzs_clip = clip_T_world(xyz.permute(0, 2, 1).float(), K.float(), E.float(), resolution[0], resolution[1]).contiguous()
            outputs, _ = nvdiffrast.torch.rasterize(self.rasterize_context, xyzs_clip, self.faces.type(torch.int32),
                                                [resolution[0], resolution[1]])
            triangle_ids = outputs[..., -1].long() - 1

        for i, triangle_id in enumerate(triangle_ids):
            visible_faces = triangle_id[triangle_id >= 0].unique()

            visible_vertices = self.faces[visible_faces].unique()

            visibility_map[i, visible_vertices] = True

        return visibility_map

        


class AnimationNet(nn.Module):
    def __init__(self, args):
        super(AnimationNet, self).__init__()
        self.args = args
        self.smpl_model = SMPL(
            model_path=self.args.smplx_model_path,
            # gender=self.args.gender,
            batch_size=1,
            create_global_orient=False,
            create_betas=False,
            create_body_pose=False,
            create_transl=False,
        ).to(self.args.device)

        # subdivide smpl model
        self.smpl_model = subdivide_smpl_model(self.smpl_model, n_iter=1, SMPL_PATH=self.args.smplx_model_path)

        self.lbs_weights = self.smpl_model.lbs_weights # [N, 24]
        self.parents = self.smpl_model.parents # [24]
        self.posedirs = self.smpl_model.posedirs # [24, 3]
        self.J_regressor = self.smpl_model.J_regressor # [24, 6890]
        self.v_template = self.smpl_model.v_template # [6890, 3]
        self.joints = torch.einsum('bik,ji->bjk', [self.v_template.unsqueeze(0), self.J_regressor]) # [1, 24, 3]
        # self.edges = torch.tensor(get_edges_from_faces(self.smpl_model.faces)).to(self.args.device) # [2, 20664]
        
        # self.lbs_weights = self.smpl_model.lbs_weights # [N, 24]
        # self.parents = self.smpl_model.parents # [24]
        # self.J_regressor = self.smpl_model.J_regressor # [24, 6890]
        # self.v_template = self.smpl_model.v_template # [6890, 3]
        # self.joints = torch.einsum('bik,ji->bjk', [self.v_template.unsqueeze(0), self.J_regressor]) # [1, 24, 3]
        if self.args.deform:
            self.deformer = GaussianDeformer(self.args)
        else:
            self.deforme_none = simple_stack

    def forward(self, gaussianses, poses, cam_params, is_train=True):
        B, N_gaussians, _ = gaussianses['xyz'].shape
        B, N_poses, _ = poses['body_pose'].shape

        # print(f"gaussianses['xyz'].shape: {gaussianses['xyz'].shape}")
        # print(f"poses['body_pose'].shape: {poses['body_pose'].shape}")

        all_rendered_images = []

        # if not deform, just use lbs
        if not self.args.deform:
            # LBS 
            transformed_gaussians = self.lbs_transform(gaussianses, poses) #B -> B*N_poses
        else:
            transformed_gaussians = gaussianses
        
        # Render gaussians
        image = render_batch(transformed_gaussians, cam_params['intrinsic'], cam_params['extrinsic'], self.args)
        
        return image
    
    def get_transformation_matrix(self, gaussians, poses):
        '''
        Here we only calculate the transformation matrix T, which is used as input of the deformer.

        '''
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
        # W = self.lbs_weights.unsqueeze(dim=0).expand([B, -1, -1])#  + lbs_offset
        # print(f"gaussians['lbs_weights'].shape: {gaussians['lbs_weights'].shape}")
        W = gaussians['lbs_weights'] #.unsqueeze(dim=0).expand([B, -1, -1])
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
        
        return 
        
    def lbs_transform(self, gaussians, smpl_params):
        B, N_pose, _ = smpl_params['body_pose'].shape
        body_pose = smpl_params['body_pose']
        global_trans = smpl_params['trans']

        # print(f"body_pose shape: {body_pose.shape}")
        # print(f"global_trans shape: {global_trans.shape}")

        ident = torch.eye(3, device=self.args.device)
        rot_mats = batch_rodrigues(body_pose.reshape(-1, 3)).view(
            [B, N_pose, -1, 3, 3])
        
        # print(f"rot_mats shape: {rot_mats.shape}")
        
        pose_feature = (rot_mats[:, :, 1:, :, :] - ident).view([B, N_pose, -1])
        # (B x N_pose x P) x (P, V * 3) -> B x N_pose x V x 3
        pose_offsets = torch.matmul(
            pose_feature, self.posedirs).view(B, N_pose, -1, 3)

        # print(f"pose_feature shape: {pose_feature.shape}")
        # print(f"pose_offsets shape: {pose_offsets.shape}")

        J_transformed, A = batch_rigid_transform(rot_mats.view(B*N_pose, -1, 3, 3), 
                                               self.joints.repeat(B*N_pose, 1, 1), 
                                               self.parents)
        A = A.view(B, N_pose, -1, 4, 4)
        
        # print(f"J_transformed shape: {J_transformed.shape}")
        # print(f"A shape: {A.shape}")
        
        # W = self.lbs_weights.unsqueeze(0).unsqueeze(0).expand([B, N_pose, -1, -1])
        W = gaussians['lbs_weights'].unsqueeze(1).repeat(1, N_pose, 1, 1)
        num_joints = self.joints.shape[1]
        T = torch.matmul(W, A.view(B, N_pose, num_joints, 16)) \
            .view(B, N_pose, -1, 4, 4)
        
        # print(f"W shape: {W.shape}")
        # print(f"T shape: {T.shape}")

        # Add pose offsets to gaussian
        gaussians_posed = gaussians['xyz'].unsqueeze(1).expand(-1, N_pose, -1, -1) + pose_offsets

        # print(f"gaussians_posed shape: {gaussians_posed.shape}")

        # Transform gaussian positions and rotations
        homogen_coord = torch.ones([B, N_pose, gaussians_posed.shape[2], 1], device=self.args.device)
        homogeneous_xyz = torch.cat([
            gaussians_posed,
            homogen_coord
        ], dim=-1)  # [B, N_pose, N, 4]
        transformed_xyz = torch.matmul(T, homogeneous_xyz.unsqueeze(-1))  # [B, N_pose, N, 4, 1]
        transformed_xyz = transformed_xyz.squeeze(-1)[..., :3]  # [B, N_pose, N, 3]
        
        # print(f"homogeneous_xyz shape: {homogeneous_xyz.shape}")
        # print(f"transformed_xyz shape: {transformed_xyz.shape}")
        
        rotation_matrix = T[..., :3, :3]  # [B, N_pose, N, 3, 3]
        current_quat = gaussians['rot'].unsqueeze(1).expand(-1, N_pose, -1, -1)  # [B, N_pose, N, 4]
        current_rot_mat = quaternion_to_matrix(current_quat)  # [B, N_pose, N, 3, 3]

        # print(f"rotation_matrix shape: {rotation_matrix.shape}")
        # print(f"current_quat shape: {current_quat.shape}")
        # print(f"current_rot_mat shape: {current_rot_mat.shape}")

        new_rot_mat = torch.matmul(rotation_matrix, current_rot_mat)  # [B, N_pose, N, 3, 3]
        new_quat = matrix_to_quaternion(new_rot_mat)  # [B, N_pose, N, 4]

        # print(f"new_rot_mat shape: {new_rot_mat.shape}")
        # print(f"new_quat shape: {new_quat.shape}")

        # add global translation
        transformed_xyz = transformed_xyz + global_trans.unsqueeze(2)
        
        # print(f"transformed_xyz shape: {transformed_xyz.shape}")
        # print(f"new_quat shape: {new_quat.shape}")

        expanded_gaussians = {}
        for key in gaussians.keys():
            if isinstance(gaussians[key], torch.Tensor):
                expanded_gaussians[key] = gaussians[key].unsqueeze(1).repeat(1, N_pose, 1, 1)
            else:
                expanded_gaussians[key] = gaussians[key]
        
        expanded_gaussians['xyz'] = transformed_xyz # [B*N_pose, N_gs, 3] 
        expanded_gaussians['rot'] = new_quat # [B*N_pose, N_gs, 4]
        
        return expanded_gaussians


def simple_stack(gaussians, poses):
    B = poses.shape[0]
    N = gaussians['xyz'].shape[0]

    deformed_gaussians = []
    # lbs_offsets = torch.zeros([B, N, 24], device=lbs_weights.device)

    for i in range(B):
        deformed_gs = {}
        deformed_gs['xyz'] = gaussians['xyz']
        deformed_gs['scale'] = gaussians['scale']
        deformed_gs['rot'] = gaussians['rot']
        deformed_gs['opacity'] = gaussians['opacity']
        deformed_gs['color'] = gaussians['color']
        deformed_gs['lbs_weights'] = gaussians['lbs_weights']
        deformed_gs['feats'] = gaussians['feats']

        deformed_gaussians.append(deformed_gs)

    combined_deformed_gs = {}
    for key in deformed_gaussians[0].keys():
        combined_deformed_gs[key] = torch.stack([d[key] for d in deformed_gaussians], dim=0)

    return combined_deformed_gs


def draw_gaussians(projected_gaussians, rgb_images, label='projected'):
    B, N_pose, H, W, _ = rgb_images.shape
    
    rgb_reshaped = rgb_images.clone().reshape(B*N_pose, H, W, 3)
    projected_gaussians_reshaped = projected_gaussians.clone().reshape(B*N_pose, -1, 2)
    for i in range(B*N_pose):
        img = rgb_reshaped[i]
        
        projected_points = projected_gaussians_reshaped[i].round().long()
        
        mask = (projected_points[..., 0] >= 0) & (projected_points[..., 0] < H) & \
                (projected_points[..., 1] >= 0) & (projected_points[..., 1] < W)
        
        valid_points = projected_points[mask]
        
        # 在这些点的位置画白色点
        if valid_points.shape[0] > 0:
            img[valid_points[:, 1], valid_points[:, 0]] = torch.tensor([1.0, 1.0, 1.0], device=rgb_images.device)
        
        # 保存图像
        import torchvision
        save_path = f"test_images/{i}.png"
        torchvision.utils.save_image(img.permute(2, 0, 1), save_path)

    # save original image
    # save_path = f"test_images/orig.png"
    # torchvision.utils.save_image(rgb_images[0, i].permute(2, 0, 1), save_path)
