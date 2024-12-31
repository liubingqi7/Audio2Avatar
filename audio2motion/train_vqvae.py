from utils import config
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
from datasets.build_vocab import Vocab


if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.1'
    os.environ["MASTER_PORT"]='8675'
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    args = config.parse_args()
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    train_data = __import__(f"datasets.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.batch_size,  
        shuffle=False if args.ddp else True,  
        num_workers=args.loader_workers,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(train_data) if args.ddp else None, 
    )

    for batch in train_loader:
        # print(batch["pose"].shape)
        print(batch["audio"].shape)
        # print(batch["facial"].shape)
        # print(batch["beta"].shape)

        break

