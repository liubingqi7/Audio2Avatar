from pathlib import Path
import numpy as np
import torch
import smplx
from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
import os

class SMPLGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, disable_smpl_static_offset=False, not_finetune_smpl_params=False):
        """
        Initialize SMPLGaussianModel
        Args:
            sh_degree: degree of spherical harmonics
            disable_smpl_static_offset: whether to disable static offset for SMPL model
            not_finetune_smpl_params: whether to freeze SMPL parameters during training
        """
        super().__init__(sh_degree)

        self.disable_smpl_static_offset = disable_smpl_static_offset
        self.not_finetune_smpl_params = not_finetune_smpl_params

        # Initialize SMPL model
        self.smpl_model = smplx.SMPL(
            model_path='/media/qizhu/Expansion/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
            gender='neutral',
            batch_size=1
        ).cuda()
        self.smpl_param = None
        self.smpl_param_orig = None

        # Initialize binding between gaussians and SMPL mesh faces
        if self.binding is None:
            self.binding = torch.arange(len(self.smpl_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.smpl_model.faces), dtype=torch.int32).cuda()

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        """
        Load mesh data and initialize SMPL parameters
        """
        if self.smpl_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1
            num_verts = self.smpl_model.v_template.shape[0]

            # Initialize static offset if enabled
            if not self.disable_smpl_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - static_offset.shape[0]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            # Initialize SMPL parameters
            self.smpl_param = {
                'betas': torch.from_numpy(meshes[0]['betas']),
                'body_pose': torch.zeros([T, 69]),
                'global_orient': torch.zeros([T, 3]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            # Load pose parameters for each timestep
            for i, mesh in pose_meshes.items():
                self.smpl_param['body_pose'][i] = torch.from_numpy(mesh['body_pose'])
                self.smpl_param['global_orient'][i] = torch.from_numpy(mesh['global_orient'])
                self.smpl_param['translation'][i] = torch.from_numpy(mesh['translation'])

            # Move parameters to GPU
            for k, v in self.smpl_param.items():
                self.smpl_param[k] = v.float().cuda()
            
            # Store original parameters
            self.smpl_param_orig = {k: v.clone() for k, v in self.smpl_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass

    def select_mesh_by_timestep(self, timestep, original=False):
        """
        Update mesh properties for a specific timestep
        Args:
            timestep: current timestep
            original: whether to use original parameters
        """
        self.timestep = timestep
        smpl_param = self.smpl_param_orig if original and self.smpl_param_orig is not None else self.smpl_param
        
        # Forward SMPL model
        verts = self.smpl_model(
            betas=smpl_param['betas'],
            body_pose=smpl_param['body_pose'][[timestep]],
            global_orient=smpl_param['global_orient'][[timestep]],
            transl=smpl_param['translation'][[timestep]],
            return_verts=True
        ).vertices

        # Add offsets
        verts = verts + smpl_param['static_offset'] + smpl_param['dynamic_offset'][[timestep]]

        # verts[..., 1] = -verts[..., 1]

        self.update_mesh_properties(verts)

    def update_mesh_properties(self, verts):
        """
        Update mesh properties based on new vertices
        Args:
            verts: vertex positions [B, V, 3]
        """
        faces = self.smpl_model.faces
        faces = faces.astype(np.int64)
        faces = torch.from_numpy(faces)
        triangles = verts[:, faces]

        # Compute face centers
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # Compute face orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            verts.squeeze(0), faces, return_scale=True
        )
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))

        # Store mesh data
        self.verts = verts
        self.faces = faces

    def training_setup(self, training_args):
        """
        Setup training parameters and optimizers
        """
        super().training_setup(training_args)

        if self.not_finetune_smpl_params:
            print("not finetuning smpl params")
            return

        # Setup pose parameters
        self.smpl_param['body_pose'].requires_grad = True
        self.smpl_param['global_orient'].requires_grad = True
        params = [
            self.smpl_param['body_pose'],
            self.smpl_param['global_orient'],
        ]
        param_pose = {'params': params, 'lr': training_args.smpl_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # Setup translation parameters
        self.smpl_param['translation'].requires_grad = True
        param_trans = {'params': [self.smpl_param['translation']], 'lr': training_args.smpl_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)

    def save_ply(self, path):
        """Save model and SMPL parameters"""
        super().save_ply(path)

        # Save SMPL parameters
        npz_path = Path(path).parent / "smpl_param.npz"
        smpl_param = {k: v.cpu().numpy() for k, v in self.smpl_param.items()}
        np.savez(str(npz_path), **smpl_param)

    def save_absolute_ply(self, path):
        """Save absolute ply file"""
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self.get_scaling.detach().cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        if self.binding is not None:
            binding = self.binding.detach().cpu().numpy()
            attributes = np.concatenate((attributes, binding[:, None]), axis=1)
            
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, **kwargs):
        """Load model and SMPL parameters"""
        super().load_ply(path)

        if not kwargs['has_target']:
            # Load finetuned SMPL parameters
            npz_path = Path(path).parent / "smpl_param.npz"
            smpl_param = np.load(str(npz_path))
            smpl_param = {k: torch.from_numpy(v).cuda() for k, v in smpl_param.items()}

            self.smpl_param = smpl_param
            self.num_timesteps = self.smpl_param['body_pose'].shape[0]
