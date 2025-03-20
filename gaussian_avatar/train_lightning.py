import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from models.lightning_wrapper import GaussianAvatar
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader, RandomSampler
from datasets.dataset_video import VideoDataset
from datasets.dataset_thuman import BaseDataset
from datasets.dataset_zjumocap import ZJUMocapDataset
from utils.data_utils import collate_fn, collate_fn_zjumocap, collate_fn_thuman
from lightning.pytorch.callbacks import ModelCheckpoint

seed_everything(42, workers=True)

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
    parser.add_argument('--num_iters', type=int, default=2,
                        help='Number of iterations for training the gaussian net.')
    parser.add_argument('--deform', action='store_true', help='Whether to use debug mode')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training the gaussian net.')
    parser.add_argument('--total_steps', type=int, default=200000, help='Total steps for training the gaussian net.')
    parser.add_argument('--dataset', type=str, default='zjumocap', help='Dataset to use for training the gaussian net.')
    parser.add_argument('--n_input_frames', type=int, default=4, help='Number of input frames for training the gaussian net.')
    parser.add_argument('--n_test_frames', type=int, default=4, help='Number of test frames for training the gaussian net.')
    
    args = parser.parse_args()

    return args

def main():
    args = setup_parser()
    logger = prepare_output_and_logger(args)

    # 根据dataset参数选择不同的数据集
    if args.dataset == "thuman":
        dataset = BaseDataset(
            dataset_root="/home/liubingqi/work/liubingqi/thuman2.0/view5_train",
            scene_list=["/home/liubingqi/work/liubingqi/thuman2.0/train.json"],
            use_smplx=True,
            smpl_dir="/home/liubingqi/work/liubingqi/THuman/THuman2.0_smpl",
            n_input_frames=args.n_input_frames,
        )
        collate_function = collate_fn_thuman
        sampler = None
    elif args.dataset == "zjumocap":
        dataset = ZJUMocapDataset(
            dataset_root='/home/liubingqi/work/Audio2Avatar/gaussian_avatar/data/zju_mocap',
            smpl_path='/home/liubingqi/work/liubingqi/SMPL_SMPLX/SMPL_models/smpl/SMPL_NEUTRAL.pkl',
            n_input_frames=args.n_input_frames,
            n_test_frames=args.n_test_frames,
        )
        collate_function = collate_fn_zjumocap
        sampler = RandomSampler(dataset, replacement=True, num_samples=1000)

    else:
        dataset = VideoDataset(args)
        collate_function = collate_fn

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_function, 
        num_workers=89,
        sampler=sampler
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='{epoch}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train/loss',
        mode='min',
        save_last=True
    )

    model = GaussianAvatar(args)

    if args.use_ckpt:
        if args.ckpt_path is not None:
            print(f"Loading pretrained weights from {args.ckpt_path}...")
            checkpoint = torch.load(args.ckpt_path)
            # If it's a lightning checkpoint, we need to extract the model weights
            if 'state_dict' in checkpoint:
                # Extract GaussianNet weights from lightning checkpoint
                gaussian_net_state_dict = {k.replace('gaussian_net.', ''): v for k, v in checkpoint['state_dict'].items() 
                                          if k.startswith('gaussian_net.')}
                model.gaussian_net.load_state_dict(gaussian_net_state_dict)
                print("Successfully loaded GaussianNet weights")
            else:
                # Directly load model weights
                model.gaussian_net.load_state_dict(checkpoint)
                print("Successfully loaded model weights")

    trainer = L.Trainer(
        # default_root_dir=args.ckpt_path,
        # max_epochs=args.num_epochs,
        max_steps=args.total_steps,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        devices=1,
        # strategy='ddp',
        gradient_clip_algorithm='norm',
        gradient_clip_val=1.0,
        # fast_dev_run=True
    )

    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
