import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.lightning_wrapper import GaussianAvatar
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from datasets.dataset_video import VideoDataset
from datasets.dataset_thuman import BaseDataset
from utils.data_utils import collate_fn

def prepare_output_and_logger(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        logger = WandbLogger(project="gaussian_avatar", name=args.experiment_name)
        return logger
    else:
        return None
    
def setup_parser():
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
    parser.add_argument('--experiment_name', type=str, default='test_lightning',
                    help='Name of the experiment')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    
    args = parser.parse_args()

    return args

def main():
    args = setup_parser()
    logger = prepare_output_and_logger(args)

    # dataset = BaseDataset(
    #     dataset_root="/home/qizhu/Desktop/Work/MotionGeneration/Audio2Avatar/others/LIFe-GoM/data/thuman2.0/view5_train",
    #     scene_list=["/home/qizhu/Desktop/Work/MotionGeneration/Audio2Avatar/others/LIFe-GoM/data/thuman2.0/test.json"],
    #     use_smplx=True,
    #     smpl_dir="/media/qizhu/Expansion/THuman/THuman2.0_smpl/",
    #     n_input_frames=1,
    # )

    dataset = VideoDataset(args)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    model = GaussianAvatar(args)
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        callbacks=[],
        accelerator='gpu',
        devices=1
    )

    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
