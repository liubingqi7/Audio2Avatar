import torch
import os
from models.core.net import GaussianNet, AnimationNet
from datasets.dataset_video import VideoDataset
from argparse import ArgumentParser
from datasets.utils import collate_fn
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from models.utils.loss_utils import l1_loss, ssim

def main():
    parser = ArgumentParser()
    parser = ArgumentParser(description="Video Dataset Parameters")
    
    parser.add_argument('--data_folder', type=str, default="data/gs_data/data/m4c_processed", 
                        help='Path to the folder containing video data.')
    parser.add_argument('--clip_length', type=int, default=1, 
                        help='Length of each video clip.')
    parser.add_argument('--clip_overlap', type=int, default=0, 
                        help='Overlap between video clips. If None, defaults to half of clip_length.')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use for training.')
    parser.add_argument('--smplx_model_path', type=str, default='/media/qizhu/Expansion/SMPL_SMPLX/SMPL_models/smpl/SMPL_NEUTRAL.pkl', 
                        help='Path to the SMPL-X model.')
    parser.add_argument('--image_height', type=int, default=1080,
                        help='')
    parser.add_argument('--image_width', type=int, default=1080, 
                        help='')
    parser.add_argument('--sh_degree', type=int, default=3, 
                        help='')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--rgb', action='store_true', help='Whether to use RGB color')
    parser.add_argument('--use_ckpt', action='store_true', help='Whether to use checkpoint file')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to the checkpoint file.')
    parser.add_argument('--net_ckpt_path', type=str, default=None,
                        help='Path to the Gaussian net checkpoint file.')
    parser.add_argument('--animation_net_ckpt_path', type=str, default=None,
                        help='Path to the animation net checkpoint file.')  
    parser.add_argument('--output_dir', type=str, default='results',
                    help='Output directory for saving rendered images')
    
    args = parser.parse_args()

    dataset = VideoDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    print(f"Dataset size: {len(dataset)}")

    net = GaussianNet(args).to(args.device)
    animation_net = AnimationNet(args).to(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)

    # load model
    if args.use_ckpt:
        if args.net_ckpt_path:
            net_ckpt_path = os.path.join(args.ckpt_path, args.net_ckpt_path)    
            net.load_state_dict(torch.load(net_ckpt_path))
        if args.animation_net_ckpt_path:
            animation_net_ckpt_path = os.path.join(args.ckpt_path, args.animation_net_ckpt_path)
            animation_net.load_state_dict(torch.load(animation_net_ckpt_path))
        current_epoch = int(args.net_ckpt_path.split('_')[-1].split('.')[0])
        print(f"Loaded model from epoch {current_epoch}")
    else:
        current_epoch = 0

    optimizer = torch.optim.Adam(list(net.parameters()) + list(animation_net.parameters()), 
                               lr=args.learning_rate)
    
    # Training loop
    for epoch in range(current_epoch, args.num_epochs):
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        
        for i, data in enumerate(pbar):
            for k, v in data.smpl_parms.items():
                data.smpl_parms[k] = v.to(args.device)
            for k, v in data.cam_parms.items():
                data.cam_parms[k] = v.to(args.device)
            
            target_images = data.video.to(args.device)
            
            gaussian = net.forward(data)
            rendered_images = animation_net.forward(gaussian, data.smpl_parms, data.cam_parms).permute(0, 2, 3, 1)

            losses = {}
            losses['l1'] = l1_loss(rendered_images, target_images.squeeze(0)) * 0.8
            losses['ssim'] = (1.0 - ssim(rendered_images, target_images.squeeze(0))) * 0.2

            losses['total'] = sum([v for k, v in losses.items()])

            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            total_loss += losses['total'].item()

            # save rendered image at the end of epoch
            if i == 0 and epoch % 50 == 0:
                save_path = f"{args.output_dir}/epoch_{epoch+1}_frame_{i}.png"
                # print(f"max: {rendered_images[0].max()}, min: {rendered_images[0].min()}")
                plt.imsave(save_path, rendered_images[0].detach().cpu().numpy())
                gt_save_path = f"{args.output_dir}/epoch_{epoch+1}_frame_{i}_gt.png"
                plt.imsave(gt_save_path, target_images.squeeze(0)[0].detach().cpu().numpy())
            
            pbar.set_postfix({'l1': losses['l1'].item(),'ssim': losses['ssim'].item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")

        if epoch % 50 == 0:
            # print(f"Saving model at epoch {epoch}, {args.ckpt_path}/gaussian_net_{epoch+1}.pth")
            torch.save(net.state_dict(), f"{args.ckpt_path}/gaussian_net_{epoch+1}.pth")
            torch.save(animation_net.state_dict(), f"{args.ckpt_path}/animation_net_{epoch+1}.pth")


if __name__ == "__main__":
    main()