import torch
import lightning as L
import wandb

from models.core.net import GaussianNet, AnimationNet
from models.utils.loss_utils import l1_loss, ssim

class GaussianAvatar(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.gaussian_net = GaussianNet(args)
        self.animation_net = AnimationNet(args)
        self.lr = args.learning_rate
        # self.global_step = 0
    
    def training_step(self, batch, batch_idx):        
        for k, v in batch.smpl_parms.items():
            batch.smpl_parms[k] = v.to(self.device)
        for k, v in batch.cam_parms.items():
            batch.cam_parms[k] = v.to(self.device)
        
        target_images = batch.video.to(self.device)

        gaussian, train_gaussians = self.gaussian_net.forward(batch)
        
        # if training, render for all gaussians
        rendered_images = self.animation_net.forward(train_gaussians, batch.smpl_parms, batch.cam_parms)
        # rendered_images = self.animation_net.forward(gaussian, batch.smpl_parms, batch.cam_parms).permute(0, 2, 3, 1)

        losses = {}
        losses['l1'] = 0
        losses['ssim'] = 0

        for i, images in enumerate(rendered_images):
            i_weight = 0.8 ** (2 - i - 1) # 2 = num_iters
            losses['l1'] += i_weight * l1_loss(images.permute(0, 2, 3, 1), target_images.squeeze(0)) * 0.8
            losses['ssim'] += i_weight * (1.0 - ssim(images.permute(0, 2, 3, 1), target_images.squeeze(0))) * 0.2

            self.log(f'train/l1_{i}', l1_loss(images.permute(0, 2, 3, 1), target_images.squeeze(0)), prog_bar=False)
            self.log(f'train/ssim_{i}', 1.0 - ssim(images.permute(0, 2, 3, 1), target_images.squeeze(0)), prog_bar=False)
            self.log(f'train/loss_{i}', l1_loss(images.permute(0, 2, 3, 1), target_images.squeeze(0)) * 0.8 +  (1.0 - ssim(images.permute(0, 2, 3, 1), target_images.squeeze(0))) * 0.2, prog_bar=False)
        
        losses['total'] = sum([v for k, v in losses.items()])

        # 记录每个step的loss
        # self.log('train/step', self.global_step, prog_bar=True)
        self.log('train/loss', losses['total'], prog_bar=True)
        
        if self.global_step % 1000 == 0:
            images = target_images.squeeze(0)[0]
            for i in range(2):
                images = torch.cat([rendered_images[i].permute(0, 2, 3, 1)[0], images], dim=1)
            self.logger.experiment.log({
                f"train/comparison": wandb.Image(images.detach().cpu().numpy())
            })

        return losses['total']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer