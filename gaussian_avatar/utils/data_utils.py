import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(eq=False)
class VideoData:
    """
    Dataclass for storing video chunks
    """

    video: torch.Tensor  # B, S, H, W, C
    smpl_parms: dict
    cam_parms: dict
    width: torch.Tensor
    height: torch.Tensor
    # optional data
    segmentation: Optional[torch.Tensor] = None  # B, S, 1, H, W


def collate_fn(batch):
    """
    Collate function for video data.
    """
    video = torch.stack([b.video for b in batch], dim=0)
    smpl_parms = {k: torch.stack([b.smpl_parms[k] for b in batch], dim=0) 
                  for k in batch[0].smpl_parms.keys()}
    cam_parms = {k: torch.stack([b.cam_parms[k] for b in batch], dim=0)
                 for k in batch[0].cam_parms.keys()}
    width = torch.stack([b.width for b in batch], dim=0)
    height = torch.stack([b.height for b in batch], dim=0)

    return VideoData(
        video=video,
        smpl_parms=smpl_parms,
        cam_parms=cam_parms,
        width=width,
        height=height,
    )

def collate_fn_zjumocap(batch):
    """
    Collate function for video data.
    """
    batch_train = [item['train'] for item in batch]
    batch_test = [item['test'] for item in batch]
    
    batch_train = collate_fn(batch_train)
    batch_test = collate_fn(batch_test)

    return (batch_train, batch_test)



def collate_fn_train(batch):
    """
    Collate function for video tracks data during training.
    """
    gotit = [gotit for _, gotit in batch]
    video = torch.stack([b.video for b, _ in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b, _ in batch], dim=0)
    visibility = torch.stack([b.visibility for b, _ in batch], dim=0)
    valid = torch.stack([b.valid for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    return (
        CoTrackerData(
            video=video,
            trajectory=trajectory,
            visibility=visibility,
            valid=valid,
            seq_name=seq_name,
        ),
        gotit,
    )


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj
