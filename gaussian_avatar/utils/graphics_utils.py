#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View_torch(R, t):
    Rt = torch.zeros(4, 4, device=R.device)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

def getWorld2View2_torch(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros(4, 4, device=R.device)
    Rt[:3, :3] = R.transpose(0, 1) 
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate.to(R.device)) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt

# def getProjectionMatrix(znear, zfar, fovX, fovY, K, h, w):
#     tanHalfFovY = math.tan((fovY / 2))
#     tanHalfFovX = math.tan((fovX / 2))

    
#     if K is None:
#         # print("using colmap style")
#         top = tanHalfFovY * znear
#         bottom = -top
#         right = tanHalfFovX * znear
#         left = -right
#     else:
#         near_fx =  znear / K[0, 0]
#         near_fy = znear / K[1, 1]

#         left = - (w - K[0, 2]) * near_fx
#         right = K[0, 2] * near_fx
#         bottom = (K[1, 2] - h) * near_fy
#         top = K[1, 2] * near_fy

#     P = torch.zeros(4, 4)

#     z_sign = 1.0

#     P[0, 0] = 2.0 * znear / (right - left)
#     P[1, 1] = 2.0 * znear / (top - bottom)
#     P[0, 2] = (right + left) / (right - left)
#     P[1, 2] = (top + bottom) / (top - bottom)
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * zfar / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)
#     return P

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix_torch(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4).to(fovX.device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def focal2fov_torch(focal, pixels):
    return 2*torch.atan(pixels/(2*focal))



# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

def compute_face_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return face_normals

def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  # will have artifacts without negation

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale

def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals

def project_gaussians(gaussians, intrinsic, extrinsic):
    '''
    Project gaussians to image plane
    xyz: [B, N, 3]
    intrinsic: [3, 3]
    extrinsic: [4, 4]
    return: [B, N, 2]
    '''

    xyz = gaussians['xyz']
    rot = gaussians['rot']
    rot_mat = quaternion_to_matrix(rot)
    
    # Convert gaussian points from world coordinates to camera coordinates
    homogen_coord = torch.ones([xyz.shape[0], xyz.shape[1], 1], device=xyz.device)
    homogeneous_xyz = torch.cat([xyz, homogen_coord], dim=-1)  # [B, N, 4]
    
    # Apply extrinsic matrix

    cam_xyz = torch.matmul(extrinsic, homogeneous_xyz.transpose(-1, -2))  # [B, 4, N]
    cam_xyz = cam_xyz.transpose(-1, -2)[..., :3]  # [B, N, 3]

    cam_rot_mat = torch.matmul(extrinsic[:, :3, :3], rot_mat) # [N, 3, 3]
    cam_rot = matrix_to_quaternion(cam_rot_mat) # [N, 4]
    gaussians['rot'] = cam_rot
    
    # Apply intrinsic matrix
    projected_xy = torch.matmul(intrinsic, cam_xyz.transpose(-1, -2))  # [B, 3, N]
    projected_xy = projected_xy.transpose(-1, -2)
    projected_gaussians = projected_xy[..., :2] / projected_xy[..., 2:3]

    # gaussians['xyz'] = projected_gaussians
    return projected_gaussians

# def unproject_gaussians(gaussians, intrinsic, extrinsic):


################THUMAN################
def _subdivide(vertices,
               faces,
               face_index=None,
               vertex_attributes=None,
               return_index=False):
    """
    this function is adapted from trimesh
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
    vertex_attributes : dict
      Contains (n, d) attribute data
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    index_dict : dict
      Only returned if `return_index`, {index of
      original face : index of new faces}.
    """
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_mask]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # the new faces_subset with correct winding
    f = np.column_stack([faces_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    # turn the mask back into integer indexes
    nonzero = np.nonzero(face_mask)[0]
    # new faces start past the original faces
    # but we've removed all the faces in face_mask
    start = len(faces) - len(nonzero)
    # indexes are just offset from start
    stack = np.arange(
        start, start + len(f) * 4).reshape((-1, 4))
    # reformat into a slightly silly dict for some reason
    index_dict = {k: v for k, v in zip(nonzero, stack)}

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            attr_tris = values[faces_subset]
            if key == 'so3':
                attr_mid = np.zeros([unique.shape[0], 3], values.dtype)
            elif key == 'scale':
                edge_len = np.linalg.norm(values[edges[unique][:, 1]] - values[edges[unique][:, 0]], axis=-1)
                attr_mid = np.ones([unique.shape[0], 3], values.dtype) * edge_len[..., None]
            else:
                attr_mid = values[edges[unique]].mean(axis=1)
            new_attributes[key] = np.vstack((
                values, attr_mid))
        return new_vertices, new_faces, new_attributes, index_dict

    if return_index:
        # turn the mask back into integer indexes
        nonzero = np.nonzero(face_mask)[0]
        # new faces start past the original faces
        # but we've removed all the faces in face_mask
        start = len(faces) - len(nonzero)
        # indexes are just offset from start
        stack = np.arange(
            start, start + len(f) * 4).reshape((-1, 4))
        # reformat into a slightly silly dict for some reason
        index_dict = {k: v for k, v in zip(nonzero, stack)}

        return new_vertices, new_faces, index_dict

    return new_vertices, new_faces


def subdivide(vertices, faces, attributes, return_edges=False):
    mesh = trimesh.Trimesh(vertices, faces, vertex_attributes=attributes)
    new_vertices, new_faces, new_attributes, index_dict = _subdivide(mesh.vertices, mesh.faces, vertex_attributes=mesh.vertex_attributes)
    if return_edges:
        edges = trimesh.Trimesh(new_vertices, new_faces).edges
        return new_vertices, new_faces, new_attributes, edges, index_dict
    else:
        return new_vertices, new_faces, new_attributes, index_dict


def clip_T_world(xyzs_world, K, E, H, W):
    xyzs = torch.cat([xyzs_world, torch.ones_like(xyzs_world[..., 0:1, :])], dim=-2)
    K_expand = torch.zeros_like(E)
    fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    K_expand[:, 0, 0] = 2.0 * fx / W
    K_expand[:, 1, 1] = 2.0 * fy / H
    K_expand[:, 0, 2] = 1.0 - 2.0 * cx / W
    K_expand[:, 1, 2] = 1.0 - 2.0 * cy / H
    znear, zfar = 1e-3, 1e3
    K_expand[:, 2, 2] = -(zfar + znear) / (zfar - znear)
    K_expand[:, 3, 2] = -1.
    K_expand[:, 2, 3] = -2.0 * zfar * znear / (zfar - znear)

    # gl_transform = torch.tensor([[1., 0, 0, 0],
    #                              [0, -1., 0, 0],
    #                              [0, 0, -1., 0],
    #                              [0, 0, 0, 1.]], device=K.device)
    # gl_transform = torch.eye(4, dtype=K.dtype, device=K.device)
    # gl_transform[1, 1] = gl_transform[2, 2] = -1.

    gl_transform = torch.tensor([[1., 0, 0, 0],
                             [0, -1., 0, 0],
                             [0, 0, -1., 0],
                             [0, 0, 0, 1.]], device='cuda')
    
    return (K_expand @ gl_transform[None] @ E @ xyzs).permute(0, 2, 1)