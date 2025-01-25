import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    # print(f"rel_joints: {rel_joints.shape}")
    # print(f"joints: {joints.shape}")
    # print(f"parents: {parents.shape}")
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def get_edges_from_faces(faces):
    """
    Extract edges (as edge_index) from SMPL faces using NumPy, then convert to PyTorch Tensor.
    
    Args:
        faces (list or np.ndarray): Triangular face indices of shape (F, 3),
                                    where F is the number of faces.
    
    Returns:
        torch.Tensor: edge_index of shape (2, num_edges).
    """
    # Ensure input is a NumPy array
    if isinstance(faces, list):
        faces = np.array(faces, dtype=np.int64)
    elif not isinstance(faces, np.ndarray):
        raise TypeError("Input faces must be a list or numpy.ndarray")

    # Extract edges from triangular faces
    edges = np.vstack([
        faces[:, [0, 1]],  # Edge 1: (v0, v1)
        faces[:, [1, 2]],  # Edge 2: (v1, v2)
        faces[:, [2, 0]]   # Edge 3: (v2, v0)
    ])  # Shape: (3 * F, 2)

    # Sort edges to ensure undirected edges are consistent
    edges = np.sort(edges, axis=1)  # Sort each edge: smallest index first

    # Remove duplicate edges
    edges = np.unique(edges, axis=0)  # Remove duplicates    

    return edges.T.astype(np.int64)

# def lbs_transform(gaussians, smpl_params, index=0):
#     body_pose = smpl_params['body_pose'][:, index]
#     global_trans = smpl_params['trans'][:, index]

#     ident = torch.eye(3, device=self.args.device)
#     rot_mats = batch_rodrigues(body_pose.view(-1, 3)).view(
#         [1, -1, 3, 3])
    
#     pose_feature = (rot_mats[:, 1:, :, :] - ident).view([1, -1])
#     # (N x P) x (P, V * 3) -> N x V x 3
#     pose_offsets = torch.matmul(
#         pose_feature, self.posedirs).view(1, -1, 3)

#     J_transformed, A = batch_rigid_transform(rot_mats, self.joints, self.parents)
#     W = self.lbs_weights.unsqueeze(dim=0).expand([1, -1, -1])
#     num_joints = self.joints.shape[1]
#     T = torch.matmul(W, A.view(1, num_joints, 16)) \
#         .view(1, -1, 4, 4)
    
#     # Add pose offsets to gaussian
#     # v_posed = pose_offsets + v_shaped
#     gaussians_posed = self.gaussians['xyz'] + pose_offsets # [1, N, 3]

#     # Transform gaussian positions and rotations
#     homogen_coord = torch.ones([1, gaussians_posed.shape[1], 1], device=self.args.device)
#     homogeneous_xyz = torch.cat([
#         gaussians_posed,
#         homogen_coord
#     ], dim=-1)  # [1, N, 4]
#     transformed_xyz = torch.matmul(T, homogeneous_xyz.unsqueeze(-1))  # [1, N, 4, 1]
#     transformed_xyz = transformed_xyz.squeeze(-1)[..., :3]  # [1, N, 3]
    
#     rotation_matrix = T[..., :3, :3]  # [1, N, 3, 3]
#     current_quat = self.gaussians['rot']  # [N, 4]
#     current_rot_mat = quaternion_to_matrix(current_quat)  # [N, 3, 3]

#     new_rot_mat = torch.matmul(rotation_matrix, current_rot_mat.unsqueeze(0))  # [1, N, 3, 3]
#     new_quat = matrix_to_quaternion(new_rot_mat.squeeze(0))  # [N, 4]

#     # add global translation
#     transformed_xyz = transformed_xyz + global_trans
    
#     return transformed_xyz, new_quat