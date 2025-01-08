import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel
from scene.smpl_gaussian_model import SMPLGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
import lpips
from argparse import ArgumentParser, Namespace
from arguments import DataParams, PipelineParams, OptimizationParams
from datasets.dataset_mono import MonoDataset_train
import wandb
import smplx
import trimesh
from PIL import Image
from pathlib import Path
import numpy as np
import pickle

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
    #     sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = SMPLGaussianModel(dataset.sh_degree, dataset.disable_smpl_static_offset, dataset.not_finetune_smpl_params)
    mesh_renderer = NVDiffRenderer()
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # receive data
                net_image = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, use_original_mesh = network_gui.receive()
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        print("render mesh")
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)
                        print(f"out_dict: {out_dict}")
                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                    
                    # send data
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)
            # gaussians.save_absolute_ply(f"output_gaussians/gaussian.ply")

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # 每1000个iter保存图像
        if (iteration % 2 == 1 and iteration < 20) or iteration % 500 == 0:
            # 创建输出目录
            output_dir = Path("output_2")
            output_dir.mkdir(exist_ok=True)
            
            # 保存渲染图像
            rendered_img = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            rendered_img = rendered_img.transpose(1, 2, 0)
            rendered_img = Image.fromarray(rendered_img)
            rendered_img.save(output_dir / f"rendered_{iteration:06d}.png")
            
            # 保存原始图像
            gt_img = (gt_image.detach().cpu().numpy() * 255).astype(np.uint8)
            gt_img = gt_img.transpose(1, 2, 0)
            gt_img = Image.fromarray(gt_img)
            gt_img.save(output_dir / f"gt_{iteration:06d}.png")

            # # 保存当前的smpl参数
            # timestep = viewpoint_cam.timestep
            # smpl_params = {
            #     'betas': gaussians.smpl_param['betas'].detach().cpu().numpy(),
            #     'body_pose': gaussians.smpl_param['body_pose'][timestep].detach().cpu().numpy(),
            #     'global_orient': gaussians.smpl_param['global_orient'][timestep].detach().cpu().numpy(),
            #     'translation': gaussians.smpl_param['translation'][timestep].detach().cpu().numpy()
            # }
            # smpl_params['timestep'] = timestep
            # with open(output_dir / f"smpl_params_{iteration:06d}.pkl", "wb") as f:
            #     pickle.dump(smpl_params, f)

            # # 渲染当前时间步的SMPL mesh
            # if gaussians.binding is not None:
            #     # 创建输出目录
            #     mesh_dir = Path("output_1")
            #     mesh_dir.mkdir(exist_ok=True, parents=True)
                
            #     # 获取当前时间步的顶点和面片
            #     verts = gaussians.verts.detach().cpu().numpy().squeeze(0)  # (V, 3)
            #     faces = gaussians.faces.detach().cpu().numpy()  # (F, 3)
                
            #     # 创建trimesh对象并渲染成图像
            #     mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            #     scene = mesh.scene()
            #     png = scene.save_image(resolution=(800,800))
                
            #     # 将渲染结果保存为图像
            #     img = Image.open(trimesh.util.wrap_as_stream(png))
            #     img.save(mesh_dir / f"mesh_{iteration:06d}.png")
                
        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim

        if gaussians.binding != None:
            if opt.metric_xyz:
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:
                # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian
        

        losses['total'] = sum([v for k, v in losses.items()])
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in gaussians.optimizer.param_groups for p in group['params']], 
                    max_norm=1.0
                )
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create wandb logger
    wandb.init(project="gaussian-avatar", config=args)
    


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = DataParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
    