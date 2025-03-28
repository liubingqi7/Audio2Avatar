import torch
import lightning as L
import wandb
import os
import torchvision.utils as vutils
import time

from models.core.net import GaussianNet, AnimationNet
from models.utils.loss_utils import l1_loss, ssim

from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing

class GaussianAvatar(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gaussian_net = GaussianNet(args)
        self.animation_net = AnimationNet(args)
        self.lr = args.learning_rate
        self.num_iters = args.num_iters
        self.use_wandb = args.use_wandb
    
    def training_step(self, batch, batch_idx):
        # start_time = time.time()
        
        if self.args.dataset == "zjumocap" or self.args.dataset == "thuman":
            batch, test_batch = batch
            for k, v in test_batch.smpl_parms.items():
                test_batch.smpl_parms[k] = v.to(self.device)
            for k, v in test_batch.cam_parms.items():
                test_batch.cam_parms[k] = v.to(self.device)
            target_images = test_batch.video.to(self.device)

        for k, v in batch.smpl_parms.items():
            batch.smpl_parms[k] = v.to(self.device)
        for k, v in batch.cam_parms.items():
            batch.cam_parms[k] = v.to(self.device)
        
        train_images = batch.video.to(self.device)
        
        # data_prep_time = time.time() - start_time
        # print(f"数据准备用时: {data_prep_time:.4f}秒")
        
        # forward_start = time.time()
        gaussian, train_gaussians = self.gaussian_net.forward(batch)
        # forward_time = time.time() - forward_start
        # print(f"高斯网络前向传播用时: {forward_time:.4f}秒")
        
        # if training, render for all gaussians
        # render_start = time.time()
        rendered_images_train = []
        rendered_images_target = []
        for gaussians in train_gaussians:
            rendered_image, transformed_gaussians = self.animation_net.forward(gaussians, batch.smpl_parms, batch.cam_parms)
            rendered_images_train.append(rendered_image)
            if self.args.dataset == "zjumocap" or self.args.dataset == "thuman":
                rendered_image, transformed_gaussians = self.animation_net.forward(gaussians, test_batch.smpl_parms, test_batch.cam_parms)
                rendered_images_target.append(rendered_image)
        # rendered_images = self.animation_net.forward(gaussian, batch.smpl_parms, batch.cam_parms).permute(0, 2, 3, 1)
        # render_time = time.time() - render_start
        # print(f"渲染用时: {render_time:.4f}秒")

        # loss_start = time.time()
        losses = {}
        losses['l1'] = 0
        losses['ssim'] = 0
        # losses['l2'] = 0
        losses['laplacian'] = 0

        for i in range(self.num_iters):
            i_weight = 0.8 ** (self.num_iters - i - 1)
            
            losses['l1'] += i_weight * l1_loss(rendered_images_train[i], train_images) * 0.8
            losses['ssim'] += i_weight * (1.0 - ssim(rendered_images_train[i], train_images)) * 0.2

            # calculate laplacian loss for gaussian
            B, N, _ = train_gaussians[i]['xyz'].shape
            faces = self.gaussian_net.faces.to(self.device).unsqueeze(0).repeat(B, 1, 1)
            verts = train_gaussians[i]['xyz'].reshape(B, N, 3)
            
            meshes = Meshes(verts=verts, faces=faces)
            laplacian_loss = mesh_laplacian_smoothing(meshes)
                            
            losses['laplacian'] += i_weight * laplacian_loss
            self.log(f'train/laplacian_{i}', laplacian_loss, prog_bar=False)

            # add penalty for gaussian
            # small opacity and large scale are not preferred
            # opacity = train_gaussians[i]['opacity']
            # scale = train_gaussians[i]['scale']
            # penalty = 0.01 * torch.mean(opacity) + 0.01 * torch.mean(scale)
            # print(torch.mean(opacity), torch.mean(scale))
            # losses['penalty'] += i_weight * penalty
            # self.log(f'train/penalty_{i}', penalty, prog_bar=False)

            # calculate l2 loss between gaussian and original smpl vertices
            # verts_original = self.gaussian_net.v_shaped
            # l2_loss = 0.001 * torch.sum((verts - verts_original) ** 2)
            # losses['l2'] += i_weight * l2_loss
            # self.log(f'train/l2_{i}', l2_loss, prog_bar=False)
                
            self.log(f'train/l1_{i}_train', l1_loss(rendered_images_train[i], train_images), prog_bar=False)
            self.log(f'train/ssim_{i}_train', 1.0 - ssim(rendered_images_train[i], train_images), prog_bar=False)
            self.log(f'train/loss_{i}_train', l1_loss(rendered_images_train[i], train_images) * 0.8 + (1.0 - ssim(rendered_images_train[i], train_images)) * 0.2 + laplacian_loss, prog_bar=False)
            
            if self.args.dataset == "zjumocap" or self.args.dataset == "thuman":
                losses['l1'] += 0.25 * i_weight * l1_loss(rendered_images_target[i], target_images) * 0.8
                losses['ssim'] += 0.25 * i_weight * (1.0 - ssim(rendered_images_target[i], target_images)) * 0.2

                self.log(f'train/l1_{i}_target', l1_loss(rendered_images_target[i], target_images), prog_bar=False)
                self.log(f'train/ssim_{i}_target', 1.0 - ssim(rendered_images_target[i], target_images), prog_bar=False)
                self.log(f'train/loss_{i}_target', l1_loss(rendered_images_target[i], target_images) * 0.8 + (1.0 - ssim(rendered_images_target[i], target_images)) * 0.2 + laplacian_loss, prog_bar=False)
        
        losses['total'] = sum([v for k, v in losses.items()])
        # loss_time = time.time() - loss_start
        # print(f"损失计算用时: {loss_time:.4f}秒")

        # self.log('train/step', self.global_step, prog_bar=True)
        self.log('train/loss', losses['total'], prog_bar=True)
        
        # vis_start = time.time()
        if self.use_wandb and self.global_step % 1001 == 0:
            all_images_train = []
            
            for frame_idx in range(train_images.shape[1]):
                real_image_train = train_images[0, frame_idx]
                combined_train = torch.cat([rendered_images_train[-1][0, frame_idx], real_image_train], dim=1)
                all_images_train.append(combined_train)
            
            images_train = torch.cat(all_images_train, dim=0)
            
            log_dict = {
                f"train/comparison_train": wandb.Image(images_train.detach().cpu().numpy())
            }
            
            if self.args.dataset == "zjumocap" or self.args.dataset == "thuman":
                all_images_target = []
                for frame_idx in range(target_images.shape[1]):
                    real_image_target = target_images[0, frame_idx]
                    combined_target = torch.cat([rendered_images_target[-1][0, frame_idx], real_image_target], dim=1)
                    all_images_target.append(combined_target)
                
                images_target = torch.cat(all_images_target, dim=0)
                log_dict[f"train/comparison_target"] = wandb.Image(images_target.detach().cpu().numpy())
            
            self.logger.experiment.log(log_dict)

        elif self.global_step % 1001 == 0:
            all_images_train = []
            
            for frame_idx in range(train_images.shape[1]):
                real_image_train = train_images[0, frame_idx]
                combined_train = torch.cat([rendered_images_train[-1][0, frame_idx], real_image_train], dim=1)
                all_images_train.append(combined_train)
            
            images_train = torch.cat(all_images_train, dim=0)
        
            os.makedirs(os.path.join(self.args.output_dir, "train_images"), exist_ok=True)
            images_train_chw = images_train.detach().cpu().permute(2, 0, 1)
            vutils.save_image(
                images_train_chw,
                os.path.join(self.args.output_dir, f"train_images/comparison_train_{self.global_step}.png"),
                normalize=True
            )
        # vis_time = time.time() - vis_start
        # print(f"可视化用时: {vis_time:.4f}秒")
        
        # total_time = time.time() - start_time
        # print(f"总训练步骤用时: {total_time:.4f}秒")

        return losses['total']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer