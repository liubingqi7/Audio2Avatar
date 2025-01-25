import torch
from models.core.net import GaussianNet, AnimationNet
from datasets.dataset_video import VideoDataset
from argparse import ArgumentParser
from datasets.utils import collate_fn
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser = ArgumentParser(description="Video Dataset Parameters")
    
    parser.add_argument('--data_folder', type=str, default="data/gs_data/data/m4c_processed", 
                        help='Path to the folder containing video data.')
    parser.add_argument('--clip_length', type=int, default=2, 
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
    
    args = parser.parse_args()

    # Prepare dataset
    dataset = VideoDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    print(f"Dataset size: {len(dataset)}")

    # Initialize models
    net = GaussianNet(args).to(args.device)
    animation_net = AnimationNet(args).to(args.device)

    # Define optimizer
    optimizer = torch.optim.Adam(list(net.parameters()) + list(animation_net.parameters()), 
                               lr=args.learning_rate)
    
    # Define loss function
    criterion = nn.L1Loss()

    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0
        
        # Create progress bar for each epoch
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        
        for i, data in enumerate(pbar):
            # Move data to device
            for k, v in data.smpl_parms.items():
                data.smpl_parms[k] = v.to(args.device)
            for k, v in data.cam_parms.items():
                data.cam_parms[k] = v.to(args.device)
            
            # Get target images
            target_images = data.video.to(args.device)
            
            # Forward pass
            gaussian = net.forward(data)
            rendered_images = animation_net.forward(gaussian, data.smpl_parms, data.cam_parms).permute(0, 2, 3, 1)

            # Calculate loss
            loss = criterion(rendered_images, target_images.squeeze(0))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # save rendered image at the end of epoch
            if i == 0:
                save_path = f"results/epoch_{epoch+1}_frame_{i}.png"
                # print(f"max: {rendered_images[0].max()}, min: {rendered_images[0].min()}")
                plt.imsave(save_path, rendered_images[0].detach().cpu().numpy())
                gt_save_path = f"results/epoch_{epoch+1}_frame_{i}_gt.png"
                plt.imsave(gt_save_path, target_images.squeeze(0)[0].detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()