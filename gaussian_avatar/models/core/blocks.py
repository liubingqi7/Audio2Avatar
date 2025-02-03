import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
import collections
from torch import Tensor
from itertools import repeat
# from torch_geometric.nn import GCNConv
from models.utils.transform_utils import remove_outliers, MinMaxScaler
from models.core.pointtransformer_v3 import PointTransformerV3
from models.core.DGCNN import DGCNN

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode="zeros")
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class Encoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super(Encoder, self).__init__()
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2

        self.conv1 = nn.Conv2d(
            input_dim, 
            self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros"
        )
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock(self.in_planes, output_dim//2, self.norm_fn, stride=1)
        self.layer2 = ResidualBlock(output_dim//2, output_dim, self.norm_fn, stride=1)

        self.conv2 = nn.Conv2d(
            output_dim + input_dim,
            output_dim,
            kernel_size=3,
            padding=1,
            padding_mode="zeros"
        )
        self.norm2 = nn.InstanceNorm2d(output_dim)
        self.relu2 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_x = x
        
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = torch.cat([x, input_x], dim=1)
        
        x = self.relu2(self.norm2(self.conv2(x)))
        
        return x


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2

        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )

        a = _bilinear_intepolate(a)
        b = _bilinear_intepolate(b)
        c = _bilinear_intepolate(c)
        d = _bilinear_intepolate(d)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super(SimpleEncoder, self).__init__()
        
    def forward(self, x):
        return self.encoder(x)


class ResidualBlock_2(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock_2, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class UnetExtractor(nn.Module):
    def __init__(self, in_channel=3, encoder_dim=[64, 96, 128], norm_fn='group'):
        super().__init__()
        self.in_ds = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            ResidualBlock_2(32, encoder_dim[0], norm_fn=norm_fn),
            ResidualBlock_2(encoder_dim[0], encoder_dim[0], norm_fn=norm_fn)
        )
        self.res2 = nn.Sequential(
            ResidualBlock_2(encoder_dim[0], encoder_dim[1], stride=2, norm_fn=norm_fn),
            ResidualBlock_2(encoder_dim[1], encoder_dim[1], norm_fn=norm_fn)
        )
        self.res3 = nn.Sequential(
            ResidualBlock_2(encoder_dim[1], encoder_dim[2], stride=2, norm_fn=norm_fn),
            ResidualBlock_2(encoder_dim[2], encoder_dim[2], norm_fn=norm_fn),
        )

    def forward(self, x):
        x0 = self.in_ds(x)
        x1 = self.res1(x0)
        x2 = self.res2(x1)
        x3 = self.res3(x2)

        return x, x1, x2, x3
    

class GaussianUpdater(nn.Module):
    def __init__(self, input_dim=288+3+3+4+1*3+3+1, hidden_dim=128, output_color_dim=3):
        # feat_dim + pos + pos_emb + rot_emb + color_emb + scale_emb + opacity_emb
        super(GaussianUpdater, self).__init__()
        
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc_position = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.fc_rotation = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
        self.fc_color = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_color_dim)
        )
        
        self.fc_scale = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.fc_opacity = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, gaussians, x, edge_index):
        """
        Args:
            gaussians (dict): Dictionary containing gaussian parameters (xyz, rot, color, scale)
            x (Tensor): Input features of shape (N, 288) where N is number of vertices
            edge_index (Tensor): Edge connections for the graph, shape (2, num_edges)
        
        Returns:
            dict: Updated gaussian parameters including position, rotation, color and scale
        """
        # Encode all gaussian parameters into features
        pos = gaussians['xyz'] # [N, 3]
        rot = gaussians['rot'] # [N, 4] 
        color = gaussians['color'] # [N, 3]
        scale = gaussians['scale'] # [N, 3]
        opacity = gaussians['opacity'] # [N, 1]

        # Create positional encodings
        # Encode position with different frequencies for each dimension
        pos_emb = torch.cat([
            torch.sin(pos[..., 0:1] * 10.0),
            torch.sin(pos[..., 1:2] * 20.0), 
            torch.sin(pos[..., 2:3] * 30.0)
        ], dim=-1)
        rot_emb = rot
        color_emb = color
        scale_emb = scale
        opacity_emb = opacity
        # print(x.shape, pos_emb.shape, rot_emb.shape, color_emb.shape, scale_emb.shape)
        # Concatenate all encodings with input features
        x = torch.cat([
            x, 
            pos_emb,
            rot_emb,
            color_emb, 
            scale_emb,
            opacity_emb
        ], dim=-1)

        # Graph convolutions
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))
        
        # Compute residual updates for each parameter
        xyz_delta = self.fc_position(x) # Position residual
        rot_delta = self.fc_rotation(x) # Rotation residual  
        color_delta = self.fc_color(x) # Color residual
        scale_delta = self.fc_scale(x) # Scale residual
        opacity_delta = self.fc_opacity(x) # Opacity residual
        # Update gaussian parameters with residual updates
        gaussians['xyz'] = gaussians['xyz'] + xyz_delta * 0.1
        gaussians['rot'] = F.normalize(gaussians['rot'] + rot_delta, dim=-1) # Normalize quaternion
        gaussians['color'] = torch.clamp(gaussians['color'] + color_delta, 0.0, 1.0)
        gaussians['scale'] = torch.clamp(gaussians['scale'] + scale_delta * 0.1, 1e-6, 1000.0)
        gaussians['opacity'] = torch.clamp(gaussians['opacity'] + opacity_delta, 0.0, 1.0)
        
        # print the range of delta
        # print(f"xyz: {gaussians['xyz'].shape}, {gaussians['xyz'].min().item()}, {gaussians['xyz'].max().item()}")
        # print(f"xyz_delta: {xyz_delta.min().item()}, {xyz_delta.max().item()}")
        # print(f"rot_delta: {rot_delta.min().item()}, {rot_delta.max().item()}")
        # print(f"color_delta: {color_delta.min().item()}, {color_delta.max().item()}")
        # print(f"scale_delta: {scale_delta.min().item()}, {scale_delta.max().item()}")
        # print(f"opacity_delta: {opacity_delta.min().item()}, {opacity_delta.max().item()}")
        
        return gaussians
    

class MultiHeadMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, color_dim):
        super(MultiHeadMLP, self).__init__()
        
        self.xyz_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 3),
            nn.Tanh()
        )
        
        self.rot_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 4)
        )
        
        self.color_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, color_dim),
            nn.Sigmoid()
        )
        
        self.scale_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), 
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 3),
        )
        
        self.opacity_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(), 
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, x):
        xyz = self.xyz_head(x)
        rot = self.rot_head(x)
        color = self.color_head(x)
        scale = self.scale_head(x)
        opacity = self.opacity_head(x)

        # print(f'range of xyz: {xyz.min().item()}, {xyz.max().item()}')
        # print(f'range of rot: {rot.min().item()}, {rot.max().item()}')
        # print(f'range of color: {color.min().item()}, {color.max().item()}')
        # print(f'range of scale: {scale.min().item()}, {scale.max().item()}')
        # print(f'range of opacity: {opacity.min().item()}, {opacity.max().item()}')
        
        return {
            'xyz': xyz/10,
            'rot': rot,
            'color': color, 
            'scale': scale,
            'opacity': opacity
        }


class GaussianUpdater_2(nn.Module):
    '''
    Based on DGCNN. The network first encodes the gaussian parameters into features, 
    then predict new gaussian parameters with image features and gaussian features.
    

    Input:
        gaussians: dict, containing gaussian parameters (xyz, rot, color, scale, opacity)
        feats: Tensor, shape (N_gaussians, feat_dim)
    Output:
        gaussians: dict, updated gaussian parameters
    '''
    def __init__(self, args, input_dim=291+3+3+4+1*3+1, hidden_dim=128, output_color_dim=3, feat_dim=14):
        super(GaussianUpdater_2, self).__init__()
        self.args = args
        self.feat_encoder = PointTransformerV3(input_dim, enable_flash=False, dec_patch_size=[128, 128, 128, 128], enc_patch_size=[128, 128, 128, 128, 128])
        self.delta_predictor = MultiHeadMLP(input_dim+64, hidden_dim, output_color_dim)
        self.grid_resolution = 100

    def forward(self, gaussians, feats_image):
        # 1. normalize gaussian position to [-1, 1]
        gaussians = [gaussians]
        normalized_gs, scalers = self.normalized_gs(gaussians)
        offset = torch.tensor([gs['xyz'].shape[0] for gs in normalized_gs]).cumsum(0)

        feat = []
        
        for gs in normalized_gs:
            feat_list = []
            for key in gs.keys():
                if key=='xyz':
                    feat_list.append(gs[key])
                elif key == 'color':
                    feat_list.append(gs[key].view(gs[key].shape[0], -1))
                else:
                    feat_list.append(gs[key])
            feat.append(torch.cat(feat_list, dim=1)) #N, D
        feat = torch.cat(feat, dim=0) #Bx-N, D
        feat = torch.cat([feat, feats_image], dim=1)
        model_input = {
            'coord': torch.cat([gs['xyz'] for gs in normalized_gs], dim=0),
            'grid_size': torch.ones([3])*1.0/self.grid_resolution,
            'offset': offset.to(self.args.device),
            'feat': feat,
        }
        model_input['grid_coord'] = torch.floor(model_input['coord']*self.grid_resolution).int() #[0~1]/grid_resolution -> int

        # 2. predict delta parameters
        feat_delta = self.feat_encoder(model_input)['feat']
        feat_delta = torch.cat([feat_delta, feat], dim=1)
        delta_params = self.delta_predictor(feat_delta)

        # print(gaussians[0].keys())
        # 3. update gaussian parameters and unnormalize
        for gs in normalized_gs:
            for k, v in gs.items():
                # print(f"k: {k}, v.range: {v.min().item()}, {v.max().item()}")
                if k == 'color':
                    gs[k] = delta_params[k]
                else:
                    gs[k] = gs[k] + delta_params[k]

        unnormalized_gs = self.unnormalized_gs(normalized_gs, scalers)

        return unnormalized_gs[0]

    def normalized_gs(self, batch_gs):
        scalers = []
        batch_normalized_gs = []
        for gs in batch_gs:
            normalized_gs = {}
            scaler = MinMaxScaler()
            scaler.fit_transform(gs['xyz'])
            normalized_gs['xyz'] = scaler.transform(gs['xyz']) 
            normalized_gs['scale'] = gs['scale'] + torch.log(scaler.scale_)
            normalized_gs['color'] = gs['color']
            normalized_gs['opacity'] = gs['opacity']
            normalized_gs['rot'] = gs['rot']
            scalers.append(scaler)
            batch_normalized_gs.append(normalized_gs)
        return batch_normalized_gs, scalers

    def unnormalized_gs(self, batch_gs, scalers): #TODO
        batch_unnormalized_gs = []
        for gs, scaler in zip(batch_gs, scalers):
            unnormalized_gs = {}
            for key in gs:
                if key=='xyz': #The predicted gs may not contain means
                    unnormalized_gs['xyz'] = scaler.inverse_transform(gs['xyz'])
                elif key=='scale':
                    unnormalized_gs['scale'] = gs['scale'] - torch.log(scaler.scale_)
                else:
                    unnormalized_gs[key] = gs[key]
            batch_unnormalized_gs.append(unnormalized_gs)
        return  batch_unnormalized_gs


class GaussianDeformer(nn.Module):
    def __init__(self, args, color_dim=3, hidden_dim=128):
        super(GaussianDeformer, self).__init__()
        self.args = args
        self.color_dim = color_dim
        self._gaussian_dim = self.color_dim + 3 + 3 + 4 + 1  # xyz + scale + rot + opacity
        self._pose_dim = 75
        self._lbs_weights_dim = 24
        self.hidden_dim = hidden_dim
        
        # Dynamic graph feature extractor
        self.graph_encoder = DGCNN(args, input_channels=(self._gaussian_dim+self._lbs_weights_dim+self._pose_dim) * 2, output_channels=hidden_dim)
        
        # Pose feature encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(self._pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LBS weights feature encoder
        self.lbs_encoder = nn.Sequential(
            nn.Linear(self._lbs_weights_dim, hidden_dim//2),  # 24 is number of joints
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        
        # Offset and weight correction predictor
        self.offset_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 3 hidden_dims from graph, pose and lbs features
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self._gaussian_dim + self._lbs_weights_dim)  # 3D xyz offset + 24D lbs weight correction
        )

    def forward(self, gaussians, pose, lbs_weights):
        # Input:
        #   gaussians: single gaussian dict with keys ['xyz', 'scale', 'rot', 'opacity', 'color']
        #   pose: [B, pose_dim] multiple poses
        #   lbs_weights: [N, 24] single lbs weight matrix for all points
        # Output:
        #   deformed_gaussians: list of B gaussian dicts, each with same keys as input
        #   all_lbs_offsets: [B, N, 24] lbs weight corrections for each pose

        print(f"pose.shape: {pose.shape}")

        B = pose.shape[0]
        N = gaussians['xyz'].shape[0]
        # Prepare gaussian input features
        gaussian_feats = torch.cat([
            gaussians['xyz'],
            gaussians['scale'], 
            gaussians['rot'],
            gaussians['opacity'],
            gaussians['color']
        ], dim=-1)  # [N, _gaussian_dim]

        gaussian_feats = gaussian_feats.unsqueeze(0).expand(pose.shape[0], -1, -1)
        lbs_weights = lbs_weights.unsqueeze(0).expand(pose.shape[0], -1, -1)
        pose = pose.unsqueeze(1).expand(-1, gaussian_feats.shape[1], -1)

        print(gaussian_feats.shape, lbs_weights.shape, pose.shape)
        
        feats = torch.cat([gaussian_feats, lbs_weights, pose], dim=-1)
        # 1. Extract local features through dynamic graph
        graph_feats = self.graph_encoder(feats.permute(0, 2, 1)) # [N, hidden_dim]

        print(graph_feats.shape)

        offset_pred = self.offset_predictor(graph_feats.reshape(B*N, -1))
        offset_pred = offset_pred.reshape(B, N, -1)

        print(offset_pred.shape)
        # Process each pose to get multiple deformed gaussians
        deformed_gaussians = []
        all_lbs_offsets = []
        
        for i in range(B):  # Iterate through each pose
            # 2. Encode pose features
            # pose_feats = self.pose_encoder(p)  # [1, hidden_dim]
            # pose_feats = pose_feats.expand(gaussian_feats.shape[0], -1)  # [N, hidden_dim]
            
            # 3. Encode LBS weight features            
            # 4. Predict offset and weight correction
            # combined_feats = torch.cat([graph_feats, pose_feats, lbs_feats], dim=-1)
            # pred = self.offset_predictor(combined_feats)
            
            gaussians_offset = offset_pred[i, :, :self._gaussian_dim]  # [N, gaussian_dim]
            lbs_offset = offset_pred[i, :, self._gaussian_dim:]  # [N, 24]

            # 5. Update gaussian parameters for this pose
            deformed_gs = {}
            deformed_gs['xyz'] = gaussians['xyz'] + gaussians_offset[:, :3]
            deformed_gs['scale'] = gaussians['scale'] + gaussians_offset[:, 3:6]
            deformed_gs['rot'] = gaussians['rot'] + gaussians_offset[:, 6:10]
            deformed_gs['opacity'] = gaussians['opacity'] + gaussians_offset[:, 10:11]
            deformed_gs['color'] = gaussians['color'] + gaussians_offset[:, 11:]
            
            deformed_gaussians.append(deformed_gs)
            all_lbs_offsets.append(lbs_offset)

        # Stack lbs offsets to get [B, N, 24] tensor
        all_lbs_offsets = torch.stack(all_lbs_offsets)

        combined_deformed_gs = {}
        for key in deformed_gaussians[0].keys():
            combined_deformed_gs[key] = torch.stack([d[key] for d in deformed_gaussians], dim=0)
        
        return combined_deformed_gs, all_lbs_offsets


