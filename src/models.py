import torch
from torch import nn
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from main import DEVICE
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# TODO: 
# 1) try different combination of initialization and input for the convlstm
# 2) Optimize the forward methods for faster computations
# Implemented attention encoder (cross attention last hidden with feasible mask) with quantile as images and last hidden state of imu_features and simple decoder initilized with encoder output
# Implemented attention encoder (cross attention last hidden of quantile features and feasible map)with quantile as sequence and simple decoder initilized with encoder output imu feature as input
# Implemented attention encoder (cross attention last hidden of quantile features and feasible map)with quantile as sequence and simple decoder initilized with encoder output and quantiles as input

def visualize_map_features(model, map_input, sample_idx=10):
    """
    Visualize CNN-extracted map features for a specific sample.
    
    Args:
    - model: TrajectoryGeneratorEncoder instance.
    - map_input: Feasible map input tensor [N, C, H, W].
    - sample_idx: Index of the sample to visualize.

    Returns:
    - Visualization of feature maps.
    """
    model.eval()
    with torch.no_grad():
        feature_maps = model.context_cnn(map_input)  # Extract map features
        feature_maps = feature_maps[sample_idx].cpu().numpy()  # Convert to NumPy
        map_input = map_input[sample_idx].cpu().numpy()  # Convert to NumPy
    # Visualize individual feature maps
    fig, axes = plt.subplots(1, 8, figsize=(15, 5))
    for i in range(8):
        axes[i].imshow(feature_maps[i], cmap='viridis')
        axes[i].set_title(f"Feature Map {i+1}")

    plt.show()

def inspect_feature_dominance(model, imu_features, map_input):
    """
    Compare the magnitudes of IMU features and CNN-extracted map features.
    
    Args:
    - model: TrajectoryGeneratorEncoder instance.
    - imu_features: IMU input tensor [N, L, h].
    - map_input: Feasible map input tensor [N, C, H, W].

    Returns:
    - Average magnitudes of IMU and map features.
    """
    model.eval()
    with torch.no_grad():
        context_encoded = model.context_cnn(map_input)
        imu_features_expanded = imu_features[:, -1, :].unsqueeze(2).unsqueeze(3).expand(-1, -1, context_encoded.size(-2), context_encoded.size(-1))
        imu_magnitude = torch.mean(torch.abs(imu_features_expanded)).item()
        map_magnitude = torch.mean(torch.abs(context_encoded)).item()

    print(f"IMU Feature Magnitude: {imu_magnitude}")
    print(f"Map Feature Magnitude: {map_magnitude}")

# def make_mlp(dim_list, norm=None):
#     layers = []
#     if norm == 'spectral':
#         for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
#             layers.append(spectral_norm(nn.Linear(dim_in, dim_out)))
#             layers.append(nn.ReLU())
#     else:
#         for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
#             layers.append(nn.Linear(dim_in, dim_out))
#             layers.append(nn.ReLU())
#     return nn.Sequential(*layers)

def make_mlp(sizes, final_nonlinearity=False, **kw):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(spectral_norm(nn.Linear(sizes[i], sizes[i+1])))
        if i < len(sizes) - 2 or final_nonlinearity:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
    


def get_noise(shape):
    return torch.randn(*shape, device=DEVICE)


class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.h_dim = args.imu_hidden_dim
        self.output_dim = output_dim
        self.num_layers = args.lstm_layers
        self.batch_size = args.batch_size

        self.lstm = torch.nn.LSTM(self.input_dim, self.h_dim, self.num_layers, batch_first=True, dropout=args.dropout)
        self.linear1 = torch.nn.Linear(self.h_dim, self.output_dim * 5)
        self.linear2 = torch.nn.Linear(self.output_dim * 5, self.output_dim)


    def init_hidden_cell(self, hidden=None):
        """Initialize hidden and cell states."""
        if hidden == None:
            h = torch.zeros((self.num_layers, self.batch_size, self.h_dim), device=DEVICE)
        else:
            h = hidden
        c = torch.zeros((self.num_layers, self.batch_size, self.h_dim), device=DEVICE)
        return h, c

    def forward(self, input, hidden=None):
        output, self.hidden = self.lstm(input, self.init_hidden_cell())
        q_feature = self.linear1(output)
        traj_output = self.linear2(q_feature)
        return traj_output, output

class QLSTMModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(QLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.h_dim = args.imu_hidden_dim
        self.output_dim = output_dim
        self.num_layers = args.lstm_layers
        self.batch_size = args.batch_size
        self.num_quantiles = 2

        self.bilinear = torch.nn.Bilinear(self.input_dim, self.input_dim, self.input_dim * 4)
        self.lstm = torch.nn.LSTM(self.input_dim * 5, self.h_dim, self.num_layers, batch_first=True, dropout=args.dropout)
        self.linear1 = torch.nn.Linear(self.h_dim + self.input_dim * 5, self.num_quantiles * 5)
        self.linear2 = torch.nn.Linear(self.num_quantiles * 5, self.output_dim * self.num_quantiles)
        # self.mix_norm = nn.LayerNorm(self.input_dim * 5)
        # self.hidden = self.init_hidden_cell()

        
        # self.lstm = torch.nn.LSTM(self.input_dim, self.h_dim, self.num_layers, batch_first=True, dropout=args.dropout)
        # self.linear1 = torch.nn.Linear(self.h_dim, self.output_dim * 5)
        # self.linear2 = torch.nn.Linear(self.output_dim * 5, self.num_quantiles)
        # self.linear3 = torch.nn.Linear(self.output_dim * 5, self.num_quantiles)

    def init_hidden_cell(self, B, hidden=None):
        """Initialize hidden and cell states."""
        
        if hidden == None:
            h = torch.zeros((self.num_layers, B, self.h_dim), device=DEVICE)
        else:
            h = hidden
        c = torch.zeros((self.num_layers, B, self.h_dim), device=DEVICE)
        return h, c

    def forward(self, input, hidden=None):
        B = input.size(0)
        input_mix = self.bilinear(input, input)
        input_mix = torch.cat([input, input_mix], dim=2)
        # input_mix = self.mix_norm(input_mix)
        output, self.hidden = self.lstm(input_mix, self.init_hidden_cell(B))
        q_feature = torch.cat([input_mix, output], dim=2)
        q_feature = self.linear1(q_feature)
        q_feature = self.linear2(q_feature)
        q_feature = q_feature.reshape(self.batch_size, -1, 2, self.num_quantiles)  # (B,T,2,Q)
        traj_output_1, traj_output_2 = q_feature[..., 0, :], q_feature[..., 1, :]       # each (B,T,Q)

        
        # output, self.hidden = self.lstm(input, self.init_hidden_cell())
        # q_feature = self.linear1(output)
        # traj_output_1 = self.linear2(q_feature)
        # traj_output_2 = self.linear3(q_feature)
        return traj_output_1, traj_output_2, output

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, hidden):
        """ Initialize hidden and cell states"""
        # Expecting map_feature of shape [batch_size, hidden_dim, height, width]
        return hidden, torch.zeros_like(hidden)


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, imu_input_tensor, map_feature, seq_len):
        """
        imu_input_tensor: IMU feature sequence, shape [batch_size, seq_len, input_dim, height, width]
        map_feature: map feature used to initialize hidden state, shape [batch_size, hidden_dim, height, width]
        """
        # Initialize hidden and cell states
        hidden_state = [self.cell_list[i].init_hidden(imu_input_tensor) for i in range(self.num_layers)]
        cur_layer_input = map_feature

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            #output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input,
                                                 cur_state=[h, c])
                #output_inner.append(h)
            #layer_output = torch.stack(output_inner, dim=1)
            #cur_layer_input = layer_output  # For the next layer
        return h  # Output of the last ConvLSTM layer

class TrajectoryGeneratorEncoder(nn.Module):
    def __init__(self, input_channel, args):
        super(TrajectoryGeneratorEncoder, self).__init__()
        # Define input dimensions and ConvLSTM parameters
        self.imu_hidden_dim = args.imu_hidden_dim
        self.map_size = args.map_size
        self.context_feature_dim = args.context_feature_dim
        self.kernel_size = args.kernel_size
        self.convlstm_layers = args.convlstm_layers

        # ConvLSTM for spatio-temporal encoding
        self.conv_lstm = ConvLSTM(input_dim=self.context_feature_dim,
                                hidden_dim=self.imu_hidden_dim,
                                kernel_size=self.kernel_size,
                                num_layers=self.convlstm_layers,
                                batch_first=True)

        self.context_cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, self.context_feature_dim, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU())#,
            #nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        #)


    def forward(self, imu_lstm_features, feasible_map):
        """
        :param imu_lstm_output: LSTM output tensor of shape [N, L, h]
        :param feasible_map: Dynamic feasible map of shape [N, L, C, M, M]
        :return: Spatio-temporal encoded features [N, L, hidden_dim, M, M]
        """
        seq_len = imu_lstm_features.shape[1]
        imu_lstm_features = imu_lstm_features[:, -1, :]
        context_encoded = self.context_cnn(feasible_map)

        # Step 2: Expand the IMU LSTM output from [N, h] to [N, h, M, M]
        imu_lstm_features = imu_lstm_features.unsqueeze(2).unsqueeze(3)  # Shape: [N, h, 1, 1]
        imu_lstm_features = imu_lstm_features.expand(-1, -1, context_encoded.shape[-2], context_encoded.shape[-1])  # Shape: [N, h, M, M]
        # Step 5: Forward pass through ConvLSTM
        last_hidden_state = self.conv_lstm(imu_lstm_features, context_encoded, seq_len)

        return last_hidden_state
        
class SimpleEncoder(nn.Module):
    def __init__(self, input_channel, args):
        super(SimpleEncoder, self).__init__()
        # Define input dimensions and ConvLSTM parameters
        self.imu_hidden_dim = args.imu_hidden_dim
        self.map_size = args.map_size
        self.context_feature_dim = args.context_feature_dim
        self.kernel_size = args.kernel_size
        self.convlstm_layers = args.convlstm_layers
        self.decoder_hidden_dim = args.decoder_hidden_dim

        self.context_cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, self.context_feature_dim, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU())#,
            #nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        #)
        self.mlp = nn.Sequential(
            nn.Linear(self.imu_hidden_dim + self.context_feature_dim * int(self.map_size/4) * int(self.map_size/4), self.decoder_hidden_dim))

    def forward(self, imu_lstm_features, feasible_map):
        """
        :param imu_lstm_output: LSTM output tensor of shape [N, L, h]
        :param feasible_map: Dynamic feasible map of shape [N, L, C, M, M]
        :return: Spatio-temporal encoded features [N, L, hidden_dim, M, M]
        """
        seq_len = imu_lstm_features.shape[1]
        imu_lstm_features = imu_lstm_features.mean(dim=1)
        context_encoded = self.context_cnn(feasible_map)
        """
        # Step 2: Expand the IMU LSTM output from [N, h] to [N, h, M, M]
        imu_lstm_features = imu_lstm_features.unsqueeze(2).unsqueeze(3)  # Shape: [N, h, 1, 1]
        imu_lstm_features = imu_lstm_features.expand(-1, -1, context_encoded.shape[-2], context_encoded.shape[-1])  # Shape: [N, h, M, M]
        # Step 5: Forward pass through ConvLSTM
        last_hidden_state = self.conv_lstm(imu_lstm_features, context_encoded, seq_len)
        """
        map_flat = context_encoded.reshape(context_encoded.size(0), -1)  # Flatten to [batch_size, feature_dim]
        fused_features = torch.cat((imu_lstm_features, map_flat), dim=-1)
        last_hidden_state = self.mlp(fused_features)

        return last_hidden_state

class AttentionEncoder(nn.Module):
    def __init__(self, input_channel, args):
        super(AttentionEncoder, self).__init__()
        self.imu_hidden_dim = args.imu_hidden_dim
        self.context_feature_dim = args.context_feature_dim
        self.hidden_dim = args.decoder_hidden_dim
        self.kernel_size = args.kernel_size
        self.map_size = args.map_size

        # Linear layers to transform IMU and map features
        self.imu_to_query = nn.Linear(self.imu_hidden_dim, self.hidden_dim)
        self.map_to_key = nn.Linear(self.context_feature_dim*int(self.map_size/4)*int(self.map_size/4), self.hidden_dim)
        self.map_to_value = nn.Linear(self.context_feature_dim*int(self.map_size/4)*int(self.map_size/4), self.hidden_dim)

        # Output projection layer
        self.output_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.context_cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, self.context_feature_dim, kernel_ssecondsize=self.kernel_size, stride=1, padding=1),
            nn.ReLU())

    def forward(self, imu_lstm_features, feasible_map):
        """
        Args:
        - imu_features: IMU input tensor [batch_size, seq_len, imu_dim].
        - map_features: Map feature tensor [batch_size, map_feature_dim].

        Returns:
        - Encoded representation [batch_size, seq_len, hidden_dim].
        """
        batch_size, seq_len, _ = imu_lstm_features.shape
        context_encoded = self.context_cnn(feasible_map)
        context_encoded = context_encoded.reshape(context_encoded.size(0), -1)
        # Compute query, key, and value
        query = self.imu_to_query(imu_lstm_features)  # [batch_size, seq_len, hidden_dim]
        key = self.map_to_key(context_encoded).unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        value = self.map_to_value(context_encoded).unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]

        # Compute attention weights
        attention_weights = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # Compute attention output
        attended_map_features = torch.bmm(attention_weights, value)  # [batch_size, seq_len, hidden_dim]

        # Combine IMU and attended map features
        fused_features = query + attended_map_features
        output = self.output_layer(fused_features)  # [batch_size, seq_len, hidden_dim]

        return output.mean(dim=1)



# def summary(t):
#     return {'mean': t.mean().item(),
#             'std' : t.std().item(),
#             'min' : t.min().item(),
#             'max' : t.max().item()}
# class AttnTracker:
#     """Collects statistics for one forward pass."""
#     def __init__(self):
#         self.q, self.k = None, None
#         self.logits, self.probs = None, None

#     def hook(self, module, inp, out):
#         # soft-maxed attention
#         self.probs = out[1].detach().cpu()                        # (B,H,Lq,Lk)

#         # --- rebuild Q & K exactly as the layer does -------------
#         if module._qkv_same_embed_dim:
#             Wq, Wk, _ = module.in_proj_weight.chunk(3, 0)
#             bq, bk, _ = module.in_proj_bias.chunk(3)
#         else:                 # <-- rarely used path
#             Wq, bq = module.q_proj_weight, module.q_proj_bias
#             Wk, bk = module.k_proj_weight, module.k_proj_bias

#         query, key = inp[0], inp[1]                               # before proj
#         dk = module.head_dim**0.5

#         q = F.linear(query, Wq, bq) / dk                          # (B,Lq,D)
#         k = F.linear(key,   Wk, bk)                               # (B,Lk,D)
#         B, Lq, _ = q.shape
#         _, Lk, _ = k.shape
#         q = q.view(B, module.num_heads, Lq, module.head_dim)
#         k = k.view(B, module.num_heads, Lk, module.head_dim)

#         # save raw tensors (optional – big!)
#         self.q, self.k = q.cpu().detach(), k.cpu().detach()

#         # pre-softmax logits
#         self.logits = torch.einsum('bhtd,bhsd->bhts', q, k).cpu().detach() # (B,H,Lq,Lk)


class CrossAttentionEncoderImuMask(nn.Module):  # Quantiles as Images
    def __init__(self, input_channel, args):
        super(CrossAttentionEncoderImuMask, self).__init__()
        self.context_feature_dim = args.context_feature_dim 
        self.kernel_size = args.kernel_size
        self.map_size = args.map_size
        self.imu_hidden_dim = args.imu_hidden_dim
        self.batch_size = args.batch_size
        self.num_heads = 4
        self.embed_dim = 128 # self.context_feature_dim * int(self.map_size/4) * int(self.map_size/4)
        self.head_dim = self.embed_dim // self.num_heads
        self.use_map = args.use_map

        # ---------------------------------------  in __init__  -----------------------
        # self.attn_logits = None        # to store last batch for inspection
        # self.attn_probs  = None

        # Transform IMU features into Query
        # self.imu_mlp = nn.Linear(self.imu_hidden_dim, self.embed_dim)
        self.imu_mlp = nn.Sequential(
        nn.Linear(self.imu_hidden_dim, self.embed_dim),
        nn.LayerNorm(self.embed_dim),
        nn.Dropout(0.1),
        nn.ReLU(inplace=True))

        self.q_scale = nn.Parameter(torch.tensor(1.0))
        # F = (self.map_size // 4) ** 2        # number of spatial tokens
        # self.logit_bias = nn.Parameter(torch.zeros(self.num_heads, 1, F))
        # nn.init.normal_(self.logit_bias, std=0.02)
    
        # self.logit_bias = nn.Parameter(torch.zeros(self.num_heads, 1, 1))  # (H,1,1)
        # self.map_mlp = nn.Linear(1, self.embed_dim)
        # CNN for spatial feature extraction (Key, Value)
        self.context_cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),#inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),#inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=self.kernel_size, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),#inplace=True),
            nn.Conv2d(64, self.context_feature_dim, stride=1, kernel_size=1),
            nn.ReLU(inplace=True))#,#inplace=True))#,
            #nn.Flatten())
        # Cross-Attention
        self.map_mlp = nn.Linear(self.context_feature_dim+2, self.embed_dim)
        self.k_scale = nn.Parameter(torch.tensor(1.0))
    
        #self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True, dropout=0.1)
        # self.cross_attention.register_forward_pre_hook(
        #     self._inject_bias_as_mask
        # )
        # self.tracker = AttnTracker()
        # self.cross_attention.register_forward_hook(self._save_logits)

    def forward(self, imu_lstm_features, feasible_mask, t):
        B, _, H, W = feasible_mask.shape
        H4 = W4 = H // 4
        F_tokens = H4 * W4
        # Extract spatial features (Key, Value)
        spatial_features = self.context_cnn(feasible_mask)  # Shape: [batch_size, flattened_dim]
        # spatial_features = spatial_features.unsqueeze(-1)
        # attn_mask = (feasible_mask.squeeze() < 0.1)
        # attn_mask = attn_mask.view(self.batch_size, -1)
        spatial_features = spatial_features.permute(0, 2, 3, 1)   # [B, 32, 32, 16]
        spatial_features = spatial_features.reshape(B, int(self.map_size/4) * int(self.map_size/4), self.context_feature_dim)        # [B, F, C]

        device = spatial_features.device

        y, x = torch.meshgrid(
        torch.linspace(-1, 1, H4, device=device),
        torch.linspace(-1, 1, W4, device=device),
        indexing='ij')                           # ij = (row,col) order
        pos = torch.stack((x, y), dim=-1)            # [H4, W4, 2]
        pos = pos.reshape(1, F_tokens, 2)                   # [1, F, 2]
        pos    = pos.expand(B, -1, -1)      # [B, F, 2]
        spatial_features = torch.cat([spatial_features, pos], dim=-1)  # [B, F, C+2]
        
        spatial_embed = self.map_mlp(spatial_features)
        spatial_embed = F.layer_norm(spatial_embed, spatial_embed.shape[-1:])
        spatial_embed = spatial_embed * self.k_scale                  # learnable temperature

        # query = self.imu_mlp(imu_lstm_features.mean(dim=1))#.unsqueeze(1)  # Shape: [1, batch_size, feature_dim]
        # query = self.imu_mlp(imu_lstm_features.mean(dim=1, keepdim=True))
        query = self.imu_mlp(imu_lstm_features)
        query = query * self.q_scale
        # query = query.unsqueeze(1)  
        
        #spatial_features = spatial_features.unsqueeze(-1)  # Shape: [1, batch_size, feature_dim]
        # if not self.use_map:
            # attn_mask_bin = None        # MultiheadAttention accepts None
        # else:
        # breakpoint()
        
        attn_mask = (feasible_mask < 0.3).float()
        # attn_mask = F.max_pool2d(attn_mask, 2, 2)             # H/2 × W/2
        # attn_mask = F.max_pool2d(attn_mask, 2, 2)             # H/4 × W/4
        attn_mask_bin = attn_mask.reshape(B, -1).bool()              # [B, F]  → bool mask
        
        # logit_bias = self.logit_bias            # (H,1,F)
        # logit_bias = logit_bias.expand(B, -1, -1, -1) # [B,H,1,F]
        # logit_bias = logit_bias.reshape(B*self.num_heads, 1, F_tokens)  # (B·H,1,F)
        

        #att_mask = att_mask.view(self.batch_size, -1)
        # Transform IMU features into Query
        # query = query * 2.0
        #self.context_feature_dim += 2     # update once in __init__

        

        # key_norm_std = spatial_embed.norm(dim=-1).std(dim=-1).mean().item()
        # key_feat_std = spatial_embed.std(dim=1).mean().item()   # variance across F
        # print(f"std ∥kᵢ∥ across keys ≈ {key_norm_std:.4f}")
        # print(f"mean feature std across keys ≈ {key_feat_std:.4f}")
        # spatial_embed = spatial_embed * 10.0    # try 5–10 first; tweak later
        
        ######spatial_embed = spatial_embed.permute(1, 0, 2)
        # Apply cross-attention
        attn_output, attn_w = self.cross_attention(query, spatial_embed, spatial_embed)#, key_padding_mask=attn_mask_bin)
        # attn_output = attn_output.squeeze(1)

        # attn_w = attn_w + self.logit_bias            # broadcast (1,H,1,1)
        # attn_w = attn_w.softmax(dim=-1)
        # var_soft = self.attn_probs.var(dim=-1).mean().item()
        # std_log  = self.attn_logits.std().item()
        # print(f"var(softmax)={var_soft:.4e}   std(logits)={std_log:.3f}")

        # row_std = self.attn_logits.std(dim=-1)        # std per row
        # row_var = self.attn_probs.var(dim=-1)         # var per row
        # print("mean row-std(logits):", row_std.mean().item())
        # print("mean row-var(softmax):", row_var.mean().item())

        # # pick first sample, head 0 (adapt if multi-head debug needed)
        # mask = ~attn_mask_bin[0]                         # shape [F]
        # free_logits = self.attn_logits[0, 0, 0, mask]    # (src_len_free,)
        
        # print("std logits (free cells only):", free_logits.std().item())
        # print("var softmax (free cells only):",
        #       self.attn_probs[0, 0, mask].var().item())

        # examine ONE head, ONE query row, ONLY free cells
        # head  = 1
        # row   = 0
        # mask  = ~attn_mask_bin[0]                       # free cells bool mask  (F,)
        # free_logits = self.attn_logits[0, head, row, mask]   # (N_free,)
        
        # print("min/max logits in free cells:", free_logits.min().item(),
        #                                      free_logits.max().item())
        # print("pairwise max │max-min│:", (free_logits.max() - free_logits.min()).item())
        # if t > 0 and t % 20 == 0:
        #     print("‖q‖ :", summary(self.q.norm(dim=-1)))          # (B,H,Lq)
        #     print("‖k‖ :", summary(self.k.norm(dim=-1)))          # (B,H,Lk)
        #     print("logits:", summary(self.logits))
        #     print("probs :", summary(self.probs))
        #     row_std = self.logits.std(-1).mean().item()       # avg σ over rows
        #     print("row-wise logit σ:", row_std)        # expect ≈ 0.3-0.4 now
    
            # B, H, Lq, Lk = self.logits.shape
            # F_side = int(Lk**0.5)            # 32 when F = 1024
            
            # # 1️⃣ histogram of logits
            # plt.figure(); plt.hist(self.logits.flatten(), bins=100)
            # plt.title("Distribution of dot-product logits"); plt.show()
            
            # # 2️⃣ histogram of probas
            # plt.figure(); plt.hist(self.probs.flatten(), bins=100)
            # plt.title("Distribution of soft-max probabilities"); plt.show()
            
            # # 3️⃣ heat-map of a single head (batch 0, head 0)
            # plt.figure(figsize=(4,4))
            # plt.imshow(self.probs[0,0,0].view(F_side, F_side),
            #            origin='lower', interpolation='nearest')
            # plt.colorbar(); plt.title("Attention map – B0 H0"); plt.show()
            
        return attn_output, attn_mask_bin, attn_w

    # def _inject_bias_as_mask(self, module, args):
    #     """
    #     Args
    #     ----
    #     module : the MultiheadAttention instance
    #     args   : tuple(query, key, value, attn_mask, key_padding_mask, ...)
    #     Returns
    #     -------
    #     new_args : tuple(...)  with additive bias merged into attn_mask
    #     """
    #     query, key, value, *rest = args
    #     B = query.size(0)                # current batch
    #     F = key.size(1)                  # num. spatial tokens

    #     # (H,1,F) → (B·H, 1, F)   matches the batched-mask shape that
    #     # PyTorch expects when batch_first=True
    #     bias_mask = self.logit_bias      # param, already on device
    #     bias_mask = bias_mask.expand(B, -1, -1, -1)      # [B,H,1,F]
    #     bias_mask = bias_mask.reshape(B * self.num_heads, 1, F)

    #     # existing attn_mask (could be None)
    #     if rest and rest[0] is not None:
    #         attn_mask = rest[0]
    #         # broadcast existing mask if needed then add
    #         attn_mask = attn_mask + bias_mask
    #     else:
    #         attn_mask = bias_mask

    #     # rebuild the *args tuple*
    #     new_args = (query, key, value, attn_mask, *rest[1:])
    #     return new_args
        
    def _save_logits(self, module, inp, out):
        self.probs = out[1].detach()

        # reconstruct logits (pre‑softmax)
        if module._qkv_same_embed_dim:
            Wq, Wk, _ = module.in_proj_weight.chunk(3, dim=0)
            bq, bk, _ = module.in_proj_bias.chunk(3)
        else:
            Wq, bq = module.q_proj_weight, module.q_proj_bias
            Wk, bk = module.k_proj_weight, module.k_proj_bias

        query, key = inp[0], inp[1]  # (B,1,D) & (B,F,D)
        dk = module.head_dim ** 0.5
        q = F.linear(query, Wq, bq) / dk
        k = F.linear(key, Wk, bk)
        B, Lq, _ = q.shape
        self.q = q.reshape(B, module.num_heads, Lq, module.head_dim)
        self.k = k.reshape(B, module.num_heads, key.shape[1], module.head_dim)
        logits = torch.einsum('bhtd,bhsd->bhts', self.q, self.k)
        self.logits = logits.detach()

class TrajectoryGeneratorDecoder_2(nn.Module):
    def __init__(self, args):
        """
        :param input_dim: Dimensionality of the input (e.g., 2 for 2D positions).
        :param hidden_dim: Hidden size of the LSTM layer.
        :param latent_dim: Size of the latent vector for stochasticity.
        :param output_dim: Output size (2 for x, y coordinates).
        :param num_layers: Number of LSTM layers.
        """
        super(TrajectoryGeneratorDecoder_2, self).__init__()
        self.decoder_hidden_dim = args.decoder_hidden_dim
        self.output_dim = args.output_dim
        self.decoder_layers = args.decoder_layers
        self.embedding_dim = args.embedding_dim
        self.latent_dim = args.latent_dim
        #self.h_dim = args.imu_hidden_dim # (simple, convlstm, attention) encoder
        self.h_dim = 4 #((CrossAttentionImuMask) Encoder)
        # LSTM cell for step-by-step prediction
        # IMU features as input
        # self.embedding = nn.Linear(self.h_dim, self.embedding_dim)
        # self.attn_embed = nn.Sequential(
        # nn.Linear(128, self.embedding_dim),
        # nn.LayerNorm(self.embedding_dim),
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.1)
        # )
        self.embedding = nn.Sequential(nn.Linear(self.h_dim, self.embedding_dim),
        nn.LayerNorm(self.embedding_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1)
        )
        # self.h_init = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.decoder_hidden_dim),
        #     nn.Tanh()  # squashes to stable range
        # )
        # self.c_init = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.decoder_hidden_dim),
        #     nn.Tanh()  # squashes to stable range
        # )
        # encoder attention as input
        # self.lstm = nn.LSTM(self.embedding_dim, self.decoder_hidden_dim, self.decoder_layers, batch_first=True)
        # Output layer to generate 2D coordinates
        self.lstm = nn.LSTM(self.embedding_dim + 128 + args.latent_dim, self.decoder_hidden_dim, self.decoder_layers, batch_first=True)

        self.output_layer = nn.Linear(self.decoder_hidden_dim, self.output_dim)
        self.relu = torch.nn.ReLU()

    def _init_state_from_z(self, z):
        """z: [B, Z] -> (h0, c0) with shapes [num_layers, B, H]"""
        B = z.size(0)
        h0_flat = self.h_init(z)  # [B, L*H]
        c0_flat = self.c_init(z)  # [B, L*H]
        h0 = h0_flat.view(B, 1, self.decoder_hidden_dim).transpose(0, 1).contiguous()
        c0 = c0_flat.view(B, 1, self.decoder_hidden_dim).transpose(0, 1).contiguous()
        return (h0, c0)
        
    def forward(self, feature_vec, last_hidden_state, state_tuple=None):
        """
        :param initial_position: Initial position of shape [N, 2] (x, y coordinates).
        :param encoder_hidden: Hidden state features from the ConvLSTM encoder [N, hidden_dim].
        :param latent_vector: Latent vector for initialization [N, latent_dim].
        :param seq_length: The length of the sequence to generate (number of time steps).
        :return: Generated trajectory of shape [N, seq_length, 2].
        """
        # Step 2: Prepare input position and container for generated trajectory
        B, L, _ = feature_vec.shape
        # if state_tuple is None:
            # state_tuple = self._init_state_from_z(z)
        decoder_input = self.embedding(feature_vec)
        decoder_input = self.relu(decoder_input)
        # z_seq = z.unsqueeze(1).expand(B, L, -1)              # [B, T, Z]
        lstm_in = torch.cat((decoder_input, last_hidden_state), dim=-1)
        # attn = self.attn_embed(last_hidden_state).unsqueeze(1).expand(B, L, -1)
        output, state_tuple = self.lstm(lstm_in, state_tuple)

        # output, state_tuple = self.lstm(decoder_input, state_tuple)
        relative_position = self.output_layer(output)
        return relative_position


class Generator(nn.Module):
    def __init__(self, input_channel, args):
        super(Generator, self).__init__()
        self.imu_hidden_dim = args.imu_hidden_dim
        self.map_size = args.map_size
        self.latent_dim = args.latent_dim
        self.decoder_hidden_dim = args.decoder_hidden_dim
        self.batch_size = args.batch_size
        self.mlp_dim = args.mlp_dim
        self.context_feature_dim = args.context_feature_dim
        # self.dts = dts
        self.target_type = args.target_type
        self.embed_dim = 128

        # Encoder: ConvLSTM for spatio-temporal feature extraction
        #self.encoder = TrajectoryGeneratorEncoder(input_channel, args)
        #self.encoder  = SimpleEncoder(input_channel, args)
        self.encoder = CrossAttentionEncoderImuMask(input_channel, args)
        #self.encoder = AttentionEncoder(input_channel, args)

        #mlp_decoder_context_dims = [self.context_feature_dim*int(self.map_size/4)*int(self.map_size/4), self.mlp_dim, self.decoder_hidden_dim - self.latent_dim]  # (CrossAttentionImuMask) encoder
        #mlp_decoder_context_dims = [self.decoder_hidden_dim, self.mlp_dim, self.decoder_hidden_dim - self.latent_dim] # (simple, attention) encoder
        #mlp_decoder_context_dims = [self.imu_hidden_dim*int(self.map_size/4)*int(self.map_size/4), self.mlp_dim, self.decoder_hidden_dim - self.latent_dim]  # (convlstm) encoder

        mlp_decoder_context_dims = [self.embed_dim, self.mlp_dim, self.decoder_hidden_dim - self.latent_dim]  # (CrossAttentionImuMask) encoder

        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)
        # self.quant_norm = nn.LayerNorm(6)


        # Decoder full trajectory prediction
        self.decoder = TrajectoryGeneratorDecoder_2(args)

    def add_noise(self, _input):
        npeds = _input.size(0)
        seq_len = _input.size(1)
        # noise_shape = (self.latent_dim,)
        # z_decoder = get_noise(noise_shape)
        # vec = z_decoder.view(1, self.latent_dim).repeat(npeds, 1)
        vec = torch.randn(npeds, seq_len, self.latent_dim, device=_input.device)
        return torch.cat((_input, vec), dim=-1)

    def forward(self, map_mask, initial_position, Q_model, imu_data, bounds, t):
        # Pass IMU data through the quantile model to get upper and lower quantiles
        X_quantiles, Y_quantiles, feature_vector = Q_model(imu_data)
        # Get the plausible region
        #static_mask = self.get_masks(X_quantiles, Y_quantiles, map_mask, initial_position, bounds)

        # Simple encoder, convlstm encoder
        #last_hidden_state = self.encoder(feature_vector, static_mask) # (Simple, convlstm, attention) encoder
        # breakpoint()
        q_cat = torch.cat((X_quantiles, Y_quantiles), dim=-1)   # [B, T, 6]
        last_hidden_state, attn_mask, attn_w = self.encoder(feature_vector, map_mask, t) #(CrossAttentionImuMask Encoder
        last_hidden_state = self.add_noise(last_hidden_state)
        # breakpoint()
        #convlstm
        #last_hidden_state = last_hidden_state.view(last_hidden_state.size(0), -1)  # Flatten to [batch_size, hidden_dim * m * m]

        # noise_input = self.mlp_decoder_context(last_hidden_state).mean(dim=1)

        # decoder_h = self.add_noise(noise_input)
        # decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_c = torch.zeros((1, self.batch_size, self.decoder_hidden_dim), device=DEVICE)
        decoder_h = torch.zeros((1, self.batch_size, self.decoder_hidden_dim), device=DEVICE)

        state_tuple = (decoder_h, decoder_c)
        
        
        #IMU features as input
        #predicted_positions = self.decoder(feature_vector, state_tuple) # (simple, convlstm, attention) encoder
        # q_cat = torch.cat((X_quantiles, Y_quantiles), dim=-1)   # [B, T, 6]
        # B = q_cat.size(0)
        # z = torch.randn(B, self.latent_dim, device=q_cat.device)
        # state_tuple = None


        # q_cat = self.quant_norm(q_cat)

        predicted_positions = self.decoder(q_cat, last_hidden_state, state_tuple) #(CrossAttentionImuMask) Encoder
    
        return predicted_positions, attn_mask, attn_w, q_cat, feature_vector


# class MapEncoder(nn.Module):
#     def __init__(self, out_dim, in_ch=1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 3, padding=1),    nn.ReLU(inplace=True), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(),
#             nn.Linear(64, out_dim)
#         )
#     def forward(self, m):                        # m: [B,1,Hc,Wc]
#         return self.net(m)                       # [B,out_dim]

# def make_Dmlp(dims, act='relu', spectral=False):
#     layers = []
#     for i in range(len(dims)-1):
#         lin = nn.Linear(dims[i], dims[i+1])
#         if spectral:
#             lin = nn.utils.spectral_norm(lin)
#         layers.append(lin)
#         if i < len(dims)-2:
#             layers += [nn.LayerNorm(dims[i+1]), nn.Dropout(0.1)]
#             layers.append(nn.ReLU(inplace=True) if act=='relu' else nn.GELU())
#     return nn.Sequential(*layers)

# class TrajectoryDiscriminator(nn.Module):
#     def __init__(self, args):
#         super().__init__()
        
#         H, E, M = args.decoder_hidden_dim, args.embedding_dim, args.mlp_dim
#         self.traj_proj = nn.Linear(2, E)
#         self.imu_proj  = nn.Linear(args.imu_hidden_dim, E)
#         self.q_proj    = nn.Linear(4, E)
#         self.map_enc   = MapEncoder(out_dim=E)          # map -> E to match per-step embeds
#         self.lstm      = nn.LSTM(input_size=E*4, hidden_size=H, batch_first=True)
#         self.head      = make_Dmlp([H, M, 1], spectral=True)

#     def forward(self, traj_rel, q_seq, map_crop, imu_seq):
#         # Expect shapes: [B,T,D] for traj/imu/q with a shared T
#         B, T, _ = traj_rel.shape
#         et = self.traj_proj(traj_rel)                   # [B,T,E]
#         ei = self.imu_proj(imu_seq)                     # [B,T,E]
#         eq = self.q_proj(q_seq)                         # [B,T,E]
#         em = self.map_enc(map_crop)                     # [B,E]
#         em = em.unsqueeze(1).expand(B, T, -1)           # tile to [B,T,E]

#         x = torch.cat([et, ei, eq, em], dim=-1)         # [B,T,4E]

#         _, (h, _) = self.lstm(x)

#         # bidirectional last layer: concat forward/backward
#         # h_last = torch.cat([h[-2], h[-1]], dim=-1)      # [B, 2H]
#         return self.head(h)                        # [B,1]
        
class ConditionEncoder(nn.Module):
    """Encodes the quantile sequence q_cat: [B, T, 6] -> [B, h_dim]."""
    def __init__(self, cond_input_dim, embed_dim, h_dim):
        super().__init__()
        self.embed = nn.Linear(cond_input_dim, embed_dim)
        self.encoder = nn.LSTM(embed_dim, h_dim, num_layers=1, batch_first=True)

    def forward(self, cond_seq):
        # cond_seq: [B, T, cond_input_dim]  (here cond_input_dim = 6)
        B = cond_seq.size(0)
        dev = cond_seq.device
        x = self.embed(cond_seq)                                  # [B, T, embed_dim]
        h0 = torch.zeros((1, B, self.encoder.hidden_size), device=dev)
        c0 = torch.zeros((1, B, self.encoder.hidden_size), device=dev)
        _, (h, _) = self.encoder(x, (h0, c0))                     # h: [1, B, h_dim]
        return h.squeeze(0)                                       # [B, h_dim]

class MapEncoder(nn.Module):
    """Encodes a 1xHxW ego map crop to a vector."""
    def __init__(self, out_dim, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),    nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, out_dim)
        )

    def forward(self, m):  # m: [B,1,Hc,Wc]
        return self.net(m)  # [B, out_dim]
        

class DiscriminatorEncoder(nn.Module):
    def __init__(self, traj_hidden_size, traj_embed_size):
        super(DiscriminatorEncoder, self).__init__()

        self.h_dim = traj_hidden_size
        self.embedding_dim = traj_embed_size

        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1, batch_first=True)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

    def init_hidden(self, batch):
        h = torch.zeros((1, batch, self.h_dim), device=DEVICE)
        c = torch.zeros((1, batch, self.h_dim), device=DEVICE)
        return (h, c)

    def forward(self, obs_traj):
        traj_embedding = self.spatial_embedding(obs_traj)
        state = self.init_hidden(obs_traj.shape[0])
        output, state = self.encoder(traj_embedding, state)
        final_h = state[0].squeeze()
        return final_h

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, args):
        super(TrajectoryDiscriminator, self).__init__()
        self.h_dim = args.decoder_hidden_dim
        self.embedding_dim = args.embedding_dim
        self.mlp_dim = args.mlp_dim
        self.imu_hidden_dim = args.imu_hidden_dim

        self.encoder = DiscriminatorEncoder(self.h_dim, self.embedding_dim)
        self.cond_encoder = ConditionEncoder(cond_input_dim=4,
                                             embed_dim=self.embedding_dim,
                                             h_dim=self.h_dim)
        
        self.map_encoder  = MapEncoder(out_dim=self.h_dim, in_ch=1)
        # self.imu_encoder = nn.Sequential(
        # nn.Linear(self.imu_hidden_dim, self.h_dim),
        # nn.LayerNorm(self.h_dim),
        # nn.Dropout(0.1),
        # nn.ReLU(inplace=True))
        
        real_classifier_dims = [self.h_dim * 3, self.mlp_dim, 1]
        self.real_classifier = make_mlp(real_classifier_dims)#, norm='spectral')

    def forward(self, traj_rel, q_cat, feasible_map):
        final_h = self.encoder(traj_rel)
        cond_h = self.cond_encoder(q_cat)        # [B, h_dim]
        map_h  = self.map_encoder(feasible_map)  # [B, h_dim]
        # imu_h = self.imu_encoder(imu_feature).mean(dim=1)
        # breakpoint()
        scores = self.real_classifier(torch.cat([final_h, cond_h, map_h], dim=-1))#, map_h, imu_h], dim=-1))  # [B, 1]
        
        # scores = self.real_classifier(final_h)
        return scores