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
from typing import Union
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, SMPLGaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2_torch, getProjectionMatrix_torch, focal2fov_torch

SCALE_BIAS = 3.9
OPACITY_BIAS = 0.0

def render_batch(gaussians, K, E, args, bg_color=None, debug=False):
    '''
    Batch rendering for gaussian avatar
    Args:
        gaussians: dict containing gaussian parameters with batch dimension
        K: intrinsic matrix [B, T, 3, 3]
        E: extrinsic matrix [B, T, 4, 4] 
        args: configuration arguments
        bg_color: background color, default white
        debug: whether to print debug info
    Returns:
        rendered: rendered images [B*T, H, W, 3]
    '''
    B, T = E.shape[0], E.shape[1]
    
    # Flatten batch and time dimensions
    E_flat = E.reshape(-1, 4, 4)
    K_flat = K.reshape(-1, 3, 3)

    # Split gaussians into different dict
    xyzs = gaussians['xyz'].reshape(B*T, -1, 3)
    rots = gaussians['rot'].reshape(B*T, -1, 4)
    scales = gaussians['scale'].reshape(B*T, -1, 3)
    opacities = gaussians['opacity'].reshape(B*T, -1, 1)
    colors = gaussians['color'].reshape(B*T, -1, 3)

    # Render using base function
    rendered_images = []
    for i in range(B*T):
        rendered = render_one(xyzs[i], rots[i], scales[i], opacities[i], colors[i], K_flat[i], E_flat[i], args, bg_color, debug)
        rendered_images.append(rendered.permute(1, 2, 0))
    
    return torch.stack(rendered_images).reshape(B, T, args.image_height, args.image_width, 3)


# def render_one_batch(xyzs, rots, scales, opacities, colors, K, E, args, bg_color=None, debug=False):
#     '''
#     Batch rendering for gaussian avatar
#     '''
#     B, T = E.shape[0], E.shape[1]
    
#     R = E[:, :3, :3].reshape(3, 3).transpose(1, 2)
#     T = E[:, :3, 3]
    
#     znear = 0.01
#     zfar = 100.0

#     height = args.image_height
#     width = args.image_width

#     focal_length_y = K[:, 1, 1]
#     focal_length_x = K[:, 0, 0]
    
#     FovY = focal2fov_torch(focal_length_y, height)
#     FovX = focal2fov_torch(focal_length_x, width)

#     tanfovx = math.tan(FovX * 0.5)
#     tanfovy = math.tan(FovY * 0.5)



def render_one(xyzs, rots, scales, opacities, colors, K, E, args, bg_color=None, debug=False):
    '''
    Customized render for gaussian avatar
    '''
    
    # print(f"xyzs.shape: {xyzs.shape}")
    # print(f"rots.shape: {rots.shape}")
    # print(f"scales.shape: {scales.shape}")
    # print(f"opacities.shape: {opacities.shape}")
    # print(f"colors.shape: {colors.shape}")

    # calculate needed camera parameters
    extrinsics = E
    intrinsics = K
    # print(extrinsics.shape)
    # print(intrinsics.shape)

    R = extrinsics[:3, :3].reshape(3, 3).transpose(1, 0)
    T = extrinsics[:3, 3]

    znear = 0.01
    zfar = 100.0

    height = args.image_height
    width = args.image_width

    focal_length_y = intrinsics[1, 1]
    focal_length_x = intrinsics[0, 0]
    
    FovY = focal2fov_torch(focal_length_y, height)
    FovX = focal2fov_torch(focal_length_x, width)

    tanfovx = math.tan(FovX * 0.5)
    tanfovy = math.tan(FovY * 0.5)

    world_view_transform = getWorld2View2_torch(R, T).transpose(0, 1)
    projection_matrix = getProjectionMatrix_torch(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY, K=intrinsics, w=width, h=height).transpose(0,1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    if bg_color is None:
        bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # print(f"image_height: {height}")
    # print(f"image_width: {width}")
    # print(f"tanfovx: {tanfovx}")
    # print(f"tanfovy: {tanfovy}")
    # print(f"bg_color: {bg_color}")
    # print(f"scaling_modifier: {1.0}")
    # print(f"viewpoint_camera.world_view_transform: {world_view_transform}")
    # print(f"viewpoint_camera.full_proj_transform: {full_proj_transform}")
    # print(f"viewpoint_camera.camera_center: {camera_center}")
    # print(f"pc.active_sh_degree: {3}")

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=world_view_transform.cuda(),
        projmatrix=full_proj_transform.cuda(),
        sh_degree=3,
        campos=camera_center.cuda(),
        prefiltered=False,
        debug=False,
        antialiasing=True,
    )

    scales = torch.min(torch.exp(scales-SCALE_BIAS), torch.tensor(0.1, device=scales.device))
    opacities = torch.sigmoid(opacities-OPACITY_BIAS)

    if debug:
        scales = torch.ones_like(scales, device=scales.device) * 0.01
        opacities = torch.ones_like(opacities, device=opacities.device) * 0.1

    # print(f"scale range: {scales.min().item()}, {scales.max().item()}")
    # print(f"opacity range: {opacities.min().item()}, {opacities.max().item()}")
   
    # colors = None
    colors_precomp = None
    if not args.rgb:
        shs_view = colors.reshape(xyzs.shape[0], -1, 3).transpose(1, 2)
        dir_pp = (xyzs - camera_center.repeat(xyzs.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(args.sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = torch.clamp(colors, 0.0, 1.0)
        # print(f"colors_precomp.max(): {colors_precomp.max()}")
        # print(f"colors_precomp.min(): {colors_precomp.min()}")

    # print(f"xyzs.shape: {xyzs.shape}")
    # print(f"rots.shape: {rots.shape}")
    # print(f"scales.shape: {scales.shape}")
    # print(f"colors.shape: {colors.shape}")
    # print(f"opacities.shape: {opacities.shape}")

    # if debug:
    #     # rots = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(xyzs.shape[0], -1)
    #     scales = torch.ones((xyzs.shape[0], 3), device="cuda") * 0.02
    #     # colors = torch.ones((xyzs.shape[0], 3), device="cuda")
    #     opacities = torch.ones((xyzs.shape[0], 1), device="cuda") * 0.1
 
    screenspace_points = torch.zeros_like(xyzs, dtype=xyzs.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # test
    # colors_precomp = torch.cat([colors_precomp, torch.ones_like(colors_precomp[:, 0:1])], dim=-1)
    # print(colors_precomp.shape)

    rendered_image, radii, inv_depth = rasterizer(
    # rendered_image, radii = rasterizer(
        means3D = xyzs,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacities,
        scales = scales,
        rotations = rots,
        cov3D_precomp = None)
    
    # print(f"rendered_image.max(): {rendered_image.max()}")
    # print(f"rendered_image.min(): {rendered_image.min()}")

    rendered_image = rendered_image.clamp(0, 1)
    
    return rendered_image # [:-1, ...], rendered_image[-1, ...]


def render(viewpoint_camera, pc : Union[GaussianModel, SMPLGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # print all settings 
    # print(f"image_height: {viewpoint_camera.image_height}")
    # print(f"image_width: {viewpoint_camera.image_width}")
    # print(f"tanfovx: {tanfovx}")
    # print(f"tanfovy: {tanfovy}")
    # print(f"bg_color: {bg_color}")
    # print(f"scaling_modifier: {scaling_modifier}")
    # print(f"viewpoint_camera.world_view_transform: {viewpoint_camera.world_view_transform}")
    # print(f"viewpoint_camera.full_proj_transform: {viewpoint_camera.full_proj_transform}")
    # print(f"viewpoint_camera.camera_center: {viewpoint_camera.camera_center}")
    # print(f"pc.active_sh_degree: {pc.active_sh_degree}")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # print(f"shs: {shs.shape}")
    # print(f"colors_precomp: {colors_precomp.shape}")
    # print(f"opacity: {opacity[0:10]}")
    # print(f"scales: {scales[0:10]}")
    # print(f"rotations: {rotations.shape}")
    # print(f"cov3D_precomp: {cov3D_precomp.shape}")

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
