import torch
import lightning as L

from models.core.net import GaussianNet, AnimationNet
from models.utils.loss_utils import l1_loss, ssim

class GaussianAvatar(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.gaussian_net = GaussianNet(args)
        self.animation_net = AnimationNet(args)
        self.lr = args.learning_rate
    
    def training_step(self, batch, batch_idx):
        for k, v in batch.smpl_parms.items():
            batch.smpl_parms[k] = v.to(self.device)
        for k, v in batch.cam_parms.items():
            batch.cam_parms[k] = v.to(self.device)
        
        target_images = batch.video.to(self.device)

        gaussian = self.gaussian_net.forward(batch)
        rendered_images = self.animation_net.forward(gaussian, batch.smpl_parms, batch.cam_parms).permute(0, 2, 3, 1)

        losses = {}
        losses['l1'] = l1_loss(rendered_images, target_images.squeeze(0)) * 0.8
        losses['ssim'] = (1.0 - ssim(rendered_images, target_images.squeeze(0))) * 0.2

        losses['total'] = sum([v for k, v in losses.items()])

        self.log('train/loss', losses['total'], on_step=True, on_epoch=True, prog_bar=True)

        return losses['total']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        