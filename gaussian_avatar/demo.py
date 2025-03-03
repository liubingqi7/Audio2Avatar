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

    net = GaussianNet(args).to(args.device)
    animation_net = AnimationNet(args).to(args.device)

    # load model
    if args.net_ckpt_path:
        net_ckpt_path = os.path.join(args.ckpt_path, args.net_ckpt_path)    
        net.load_state_dict(torch.load(net_ckpt_path))
    if args.animation_net_ckpt_path:
        animation_net_ckpt_path = os.path.join(args.ckpt_path, args.animation_net_ckpt_path)
        animation_net.load_state_dict(torch.load(animation_net_ckpt_path))

    net.eval()
    animation_net.eval()

    # load reference images
    dataset = VideoDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            for k, v in data.smpl_parms.items():
                data.smpl_parms[k] = v.to(args.device)
            for k, v in data.cam_parms.items():
                data.cam_parms[k] = v.to(args.device)

            target_images = data.video.to(args.device)
            
            gaussian = net.forward(data)
            rendered_images = animation_net.forward(gaussian, data.smpl_parms, data.cam_parms).permute(0, 2, 3, 1)

        for j in range(rendered_images.shape[0]):
            save_path = f"{args.output_dir}/demo_frame_{i*4+j}_rendered.png"
            plt.imsave(save_path, rendered_images[j].detach().cpu().numpy())
            gt_path = f"{args.output_dir}/demo_frame_{i*4+j}_gt.png"
            plt.imsave(save_path, target_images[0, j].detach().cpu().numpy())
            print(f"Saved rendered and gt image to {save_path}")

    
    # smpl_params_list = []
    # cam_params_list = []
    # for i, batch in enumerate(dataloader):
    #     if i >= 20:
    #         break
    #     for k, v in batch.smpl_parms.items():
    #         batch.smpl_parms[k] = v.to(args.device)
    #     for k, v in batch.cam_parms.items():
    #         batch.cam_parms[k] = v.to(args.device)
    #     smpl_params_list.append(batch.smpl_parms)
    #     cam_params_list.append(batch.cam_parms)
    
    # for i, (smpl_params, cam_params) in enumerate(zip(smpl_params_list, cam_params_list)):
    #     rendered_image = animation_net.forward(gaussian, smpl_params, cam_params).permute(0, 2, 3, 1)
    #     print(rendered_image.shape)
        
    #     save_path = f"{args.output_dir}/demo_frame_{i}.png"
    #     plt.imsave(save_path, rendered_image[0].detach().cpu().numpy())
    #     print(f"Saved rendered image to {save_path}")

if __name__ == "__main__":
    main()