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
        for k, v in batch.smpl_parms.items():
            batch.smpl_parms[k] = v.to(self.device)
        for k, v in batch.cam_parms.items():
            batch.cam_parms[k] = v.to(self.device)
        
        target_images = batch.video.to(self.device)

        gaussian, train_gaussians = self.gaussian_net.forward(batch)
        
        # if training, render for all gaussians
        rendered_images = []
        for gaussians in train_gaussians:
            rendered_images.append(self.animation_net.forward(gaussians, batch.smpl_parms, batch.cam_parms))
        # rendered_images = self.animation_net.forward(gaussian, batch.smpl_parms, batch.cam_parms).permute(0, 2, 3, 1)

        losses = {}
        losses['l1'] = 0
        losses['ssim'] = 0

        for i, images in enumerate(rendered_images):
            i_weight = 0.8 ** (self.num_iters - i - 1)
            losses['l1'] += i_weight * l1_loss(images, target_images) * 0.8
            losses['ssim'] += i_weight * (1.0 - ssim(images, target_images)) * 0.2

            self.log(f'train/l1_{i}', l1_loss(images, target_images), prog_bar=False)
            self.log(f'train/ssim_{i}', 1.0 - ssim(images, target_images), prog_bar=False)
            self.log(f'train/loss_{i}', l1_loss(images, target_images) * 0.8 + (1.0 - ssim(images, target_images)) * 0.2, prog_bar=False)
        
        losses['total'] = sum([v for k, v in losses.items()])

        # self.log('train/step', self.global_step, prog_bar=True)
        self.log('train/loss', losses['total'], prog_bar=True)
        
        if self.use_wandb and self.global_step % 1000 == 0:
            images = target_images[0, 0]
            for i in range(self.num_iters):
                images = torch.cat([rendered_images[i][0, 0], images], dim=1)
            self.logger.experiment.log({
                f"train/comparison": wandb.Image(images.detach().cpu().numpy())
            })

        return losses['total']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.args.total_steps,
            pct_start=0.05,
            div_factor=25,
            final_div_factor=1000,
            three_phase=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }