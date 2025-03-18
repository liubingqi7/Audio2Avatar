import torch
import lightning as L
import wandb

from models.core.net import GaussianNet, AnimationNet
from models.utils.loss_utils import l1_loss, ssim

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
        if self.args.dataset == "zjumocap":
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
        target_images = test_batch.video.to(self.device)

        gaussian, train_gaussians = self.gaussian_net.forward(batch)
        
        # if training, render for all gaussians
        rendered_images_train = []
        rendered_images_target = []
        for gaussians in train_gaussians:
            rendered_images_train.append(self.animation_net.forward(gaussians, batch.smpl_parms, batch.cam_parms))
            rendered_images_target.append(self.animation_net.forward(gaussians, test_batch.smpl_parms, test_batch.cam_parms))
        # rendered_images = self.animation_net.forward(gaussian, batch.smpl_parms, batch.cam_parms).permute(0, 2, 3, 1)

        losses = {}
        losses['l1'] = 0
        losses['ssim'] = 0

        for i in range(self.num_iters):
            i_weight = 0.8 ** (self.num_iters - i - 1)
            
            losses['l1'] += i_weight * l1_loss(rendered_images_train[i], train_images) * 0.8
            losses['ssim'] += i_weight * (1.0 - ssim(rendered_images_train[i], train_images)) * 0.2

            self.log(f'train/l1_{i}_train', l1_loss(rendered_images_train[i], train_images), prog_bar=False)
            self.log(f'train/ssim_{i}_train', 1.0 - ssim(rendered_images_train[i], train_images), prog_bar=False)
            self.log(f'train/loss_{i}_train', l1_loss(rendered_images_train[i], train_images) * 0.8 + (1.0 - ssim(rendered_images_train[i], train_images)) * 0.2, prog_bar=False)

            losses['l1'] += i_weight * l1_loss(rendered_images_target[i], target_images) * 0.8
            losses['ssim'] += i_weight * (1.0 - ssim(rendered_images_target[i], target_images)) * 0.2

            self.log(f'train/l1_{i}_target', l1_loss(rendered_images_target[i], target_images), prog_bar=False)
            self.log(f'train/ssim_{i}_target', 1.0 - ssim(rendered_images_target[i], target_images), prog_bar=False)
            self.log(f'train/loss_{i}_target', l1_loss(rendered_images_target[i], target_images) * 0.8 + (1.0 - ssim(rendered_images_target[i], target_images)) * 0.2, prog_bar=False)
        
        losses['total'] = sum([v for k, v in losses.items()])

        # self.log('train/step', self.global_step, prog_bar=True)
        self.log('train/loss', losses['total'], prog_bar=True)
        
        if self.use_wandb and self.global_step % 1001 == 0:
            all_images_train = []
            all_images_target = []
            
            for frame_idx in range(train_images.shape[1]):
                real_image_train = train_images[0, frame_idx]
                real_image_target = target_images[0, frame_idx]
                
                combined_train = torch.cat([rendered_images_train[-1][0, frame_idx], real_image_train], dim=1)
                combined_target = torch.cat([rendered_images_target[-1][0, frame_idx], real_image_target], dim=1)
                all_images_train.append(combined_train)
                all_images_target.append(combined_target)
            
            images_train = torch.cat(all_images_train, dim=0)
            images_target = torch.cat(all_images_target, dim=0)
            self.logger.experiment.log({
                f"train/comparison_train": wandb.Image(images_train.detach().cpu().numpy()),
                f"train/comparison_target": wandb.Image(images_target.detach().cpu().numpy())
            })

        return losses['total']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer