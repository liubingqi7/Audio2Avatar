import torch
from models.core.net import GaussianNet, AnimationNet
from datasets.dataset_video import VideoDataset
from argparse import ArgumentParser
from datasets.utils import collate_fn
import matplotlib.pyplot as plt

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
    
    args = parser.parse_args()

    dataset = VideoDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    print(len(dataset))

    net = GaussianNet(args)
    animation_net = AnimationNet(args)

    for i, data in enumerate(dataloader):
        # move data to device
        for k, v in data.smpl_parms.items():
            data.smpl_parms[k] = v.to(args.device)
        for k, v in data.cam_parms.items():
            data.cam_parms[k] = v.to(args.device)
        gaussian= net.forward(data)
        rendered_image = animation_net.forward(gaussian, data.smpl_parms, data.cam_parms)

if __name__ == "__main__":
    main()