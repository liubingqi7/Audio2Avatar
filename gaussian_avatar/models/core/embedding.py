import torch

def get_3d_sincos_pos_embed(xyz, embed_dim):
    '''
    Input:
        xyz: Tensor, shape (N_gaussians, 3)
        embed_dim: int, embedding dimension
    Output:
        pos_embed: Tensor, shape (N_gaussians, embed_dim)
    '''
    num_points, dim = xyz.shape
    assert dim == 3, "Input must be 3D coordinates (x, y, z)"
    
    pos = xyz / 1000.0 * 2 * torch.pi
    sin_pos = torch.sin(pos)
    cos_pos = torch.cos(pos)
    
    pos_embed = torch.stack((sin_pos, cos_pos), dim=2).reshape(num_points, -1)
    return pos_embed
