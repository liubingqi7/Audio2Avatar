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

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from tqdm import tqdm
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch    

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    width: int
    height: int
    bg: np.array = np.array([0, 0, 0])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    point_cloud: Optional[BasicPointCloud]
    ply_path: Optional[str]
    val_cameras: list = []
    train_meshes: dict = {}
    test_meshes: dict = {}
    tgt_train_meshes: dict = {}
    tgt_test_meshes: dict = {}

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        width, height = image.size

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if 'camera_angle_x' in contents:
            fovx_shared = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            file_path = frame["file_path"]
            if extension not in frame["file_path"]:
                file_path += extension
            cam_name = os.path.join(path, file_path)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            
            if 'w' in frame and 'h' in frame:
                image = None
                width = frame['w']
                height = frame['h']
            else:
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                width, height = image.size

            if 'camera_angle_x' in frame:
                fovx = frame["camera_angle_x"]
            else:
                fovx = fovx_shared
            fovy = focal2fov(fov2focal(fovx, width), height)

            timestep = frame["timestep_index"] if 'timestep_index' in frame else None
            camera_id = frame["camera_index"] if 'camera_id' in frame else None
            
            cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, bg=bg, image=image, 
                image_path=image_path, image_name=image_name, 
                width=width, height=height, 
                timestep=timestep, camera_id=camera_id))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMeshesFromTransforms(path, transformsfile):
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        
        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            if not 'timestep_index' in frame or frame["timestep_index"] in mesh_infos:
                continue

            flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))
            mesh_infos[frame["timestep_index"]] = flame_param
    return mesh_infos

def readMeshesFromSMPL(path, smpl_path='smpl_parms.pth', split='train'):
    data_folder = os.path.join(path, split)
    smpl_path = os.path.join(path, split, smpl_path)
    smpl_data = torch.load(smpl_path)
    data_length = len(os.listdir(os.path.join(data_folder, 'images')))

    mesh_infos = {}
    for idx in tqdm(range(data_length), total=data_length):
        print(smpl_data.keys())
        break
        mesh_infos[idx] = {
            'betas': smpl_data.get('beta', None),
            'body_pose': smpl_data['body_pose'][idx, :69],
            'global_orient': smpl_data.get('global_orient', torch.zeros(3))[idx] if 'global_orient' in smpl_data else torch.zeros(3),
            'translation': smpl_data['trans'][idx],
            'static_offset': torch.zeros(6890, 3),
        }
    
    return mesh_infos

def readDynamicNerfInfo(path, white_background, eval, extension=".png", target_path=""):
    print("Reading Training Transforms")
    if target_path != "":
        train_cam_infos = readCamerasFromTransforms(target_path, "transforms_train.json", white_background, extension)
    else:
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    
    print("Reading Training Meshes")
    train_mesh_infos = readMeshesFromTransforms(path, "transforms_train.json")
    if target_path != "":
        print("Reading Target Meshes (Training Division)")
        tgt_train_mesh_infos = readMeshesFromTransforms(target_path, "transforms_train.json")
    else:
        tgt_train_mesh_infos = {}
    
    print("Reading Validation Transforms")
    if target_path != "":
        val_cam_infos = readCamerasFromTransforms(target_path, "transforms_val.json", white_background, extension)
    else:
        val_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension)
    
    print("Reading Test Transforms")
    if target_path != "":
        test_cam_infos = readCamerasFromTransforms(target_path, "transforms_test.json", white_background, extension)
    else:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    print("Reading Test Meshes")
    test_mesh_infos = readMeshesFromTransforms(path, "transforms_test.json")
    if target_path != "":
        print("Reading Target Meshes (Test Division)")
        tgt_test_mesh_infos = readMeshesFromTransforms(target_path, "transforms_test.json")
    else:
        tgt_test_mesh_infos = {}
    
    if target_path != "" or not eval:
        train_cam_infos.extend(val_cam_infos)
        val_cam_infos = []
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        train_mesh_infos.update(test_mesh_infos)
        test_mesh_infos = {}

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           train_meshes=train_mesh_infos,
                           test_meshes=test_mesh_infos,
                           tgt_train_meshes=tgt_train_mesh_infos,
                           tgt_test_meshes=tgt_test_mesh_infos)
    return scene_info

def readCameraFromMonoDataset(data_path, split='train', white_background=True):
    cam_path = os.path.join(data_path, split, 'cam_parms.npz')
    cam_npy = np.load(cam_path)

    extr_npy = cam_npy['extrinsic']
    intr_npy = cam_npy['intrinsic']
    R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
    T = np.array([extr_npy[:3, 3]], np.float32)
    intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)

    data_folder = os.path.join(data_path, split)
    smpl_path = os.path.join(data_path, split, 'smpl_parms.pth')
    name_list = []
    for index, img in enumerate(sorted(os.listdir(os.path.join(data_folder, 'images')))):
        base_name = img.split('.')[0]
        name_list.append((index, base_name)) 
    image_fix = os.listdir(os.path.join(data_folder, 'images'))[0].split('.')[-1]   
    mask_fix = os.listdir(os.path.join(data_folder, 'masks'))[0].split('.')[-1]
    data_length = len(name_list)
    bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
    
    cam_infos = []
    for idx in tqdm(range(data_length), total=data_length):
        pose_idx, name_idx = name_list[idx]
        image_path = os.path.join(data_folder, 'images' ,name_idx + '.' + image_fix)
        mask_path = os.path.join(data_folder, 'masks', name_idx + '.' + mask_fix)
        # cam_path = os.path.join(data_folder, 'cam_parms', name_idx + '.npz')

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        image = Image.open(image_path)
        width, height = image.size

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        mask = np.array(Image.open(mask_path))

        if len(mask.shape) <3:
            mask = mask[...,None]

        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        color_img = image * mask + (1 - mask) * 255
        image = Image.fromarray(np.array(color_img, dtype=np.byte), "RGB")

        cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, bg=bg, image=image, 
                image_path=image_path, image_name=name_idx, 
                width=width, height=height, 
                timestep=idx, camera_id=idx))
        
    return cam_infos

def readSMPLMeshes(data_path, split='train'):
    """
    Read SMPL parameters from a specific split (train/test)
    Args:
        data_path: path to dataset root
        split: 'train' or 'test'
    Returns:
        mesh_infos: dict containing SMPL parameters for each timestep
    """
    # Load SMPL parameters from .pth file
    smpl_path = os.path.join(data_path, split, 'smpl_parms.pth')
    print(f"Loading SMPL data from: {smpl_path}")
    smpl_data = torch.load(smpl_path)

    smpl_numpy = {}
    for key, value in smpl_data.items():
        if torch.is_tensor(value):
            smpl_numpy[key] = value.detach().cpu().numpy()
        else:
            smpl_numpy[key] = value
    
    # Get sequence length from number of images
    data_length = len(os.listdir(os.path.join(data_path, split, 'images')))
    print(f"Found {data_length} frames in {split} split")
    
    mesh_infos = {}
    for idx in range(data_length):
        full_pose = smpl_numpy['body_pose'][idx]
        global_orient = full_pose[:3]
        body_pose = full_pose[3:]
        
        mesh_infos[idx] = {
            'betas': smpl_numpy.get('beta', None),
            'body_pose': body_pose,
            'global_orient': global_orient,  
            'translation': smpl_numpy['trans'][idx],
            'static_offset': np.zeros((6890, 3)),
        }
    
    return mesh_infos

def readMonoDatasetInfo(path, white_background, eval=False):
    """
    Read only mesh information from monocular dataset
    Args:
        path: path to dataset root
        eval: whether in evaluation mode
    Returns:
        scene_info: SceneInfo object containing only mesh information
    """
    print("Reading Training Meshes")
    train_mesh_infos = readSMPLMeshes(path, 'train')
    
    print("Reading Test Meshes")
    test_mesh_infos = readSMPLMeshes(path, 'test')

    print("Reading Training Cameras")
    train_cam_infos = readCameraFromMonoDataset(path, 'train')

    print("Reading Test Cameras")
    test_cam_infos = readCameraFromMonoDataset(path, 'test')
    
    # If not in eval mode, merge test into train
    if not eval:
        train_mesh_infos.update(test_mesh_infos)
        test_mesh_infos = {}
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Create scene info with only mesh data
    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           val_cameras=[],
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           train_meshes=train_mesh_infos,
                           test_meshes=test_mesh_infos,
                           tgt_train_meshes={},
                           tgt_test_meshes={})
    
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "DynamicNerf" : readDynamicNerfInfo,
    "Blender" : readNerfSyntheticInfo,
    "MonoDataset": readMonoDatasetInfo,
}