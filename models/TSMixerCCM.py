import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Rev_in import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        print("\nModel initialization:")
        print(f"seq_len: {configs.seq_len}, pred_len: {configs.pred_len}")
        print(f"channels: {configs.enc_in}, hidden_size: {configs.hidden_size}")
        print(f"num_clusters: {configs.num_clusters}, num_blocks: {configs.num_blocks}")

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.hidden_size = configs.hidden_size
        self.num_clusters = configs.num_clusters
        self.num_blocks = configs.num_blocks

        self.rev_norm = RevIN(self.channels, affine=configs.affine)

        self.channel_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.cluster_embeds = nn.Parameter(torch.randn(self.num_clusters, self.hidden_size))

        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size)

        self.mixer_block = MixerBlock(self.channels, self.hidden_size,
                                      self.seq_len, configs.dropout,
                                      configs.activation, configs.single_layer_mixer,
                                      self.num_blocks)

        # New output projection layers for each cluster
        self.output_projections = nn.ModuleList([
            nn.Linear(self.hidden_size, self.pred_len) for _ in range(self.channels)
        ])

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        print("\nModel forward:")
        print(f"Input shape: {x.shape}")
        
        x = self.rev_norm(x, 'norm')
        print(f"After rev_norm shape: {x.shape}")
        
        x = x.transpose(1, 2)
        print(f"After transpose shape: {x.shape}")
        
        h_i = self.channel_mlp(x)
        print(f"After channel_mlp shape: {h_i.shape}")
        # x = self.rev_norm(x, 'norm')  # Normalize input
        # x = x.transpose(1, 2)  # [Batch, Channel, Input length]
        # h_i = self.channel_mlp(x)  # Channel embeddings via MLP [Batch, Channel, hidden_size]

        # Compute clustering probability matrix P
        c_k_norm = F.normalize(self.cluster_embeds, dim=-1)
        h_i_norm = F.normalize(h_i, dim=-1)
        p_ik = F.softmax(torch.matmul(h_i_norm, c_k_norm.t()), dim=-1)  # [Batch, Channel, K]

        # Sample Clustering Membership Matrix M
        M = torch.bernoulli(p_ik)  # [Batch, Channel, K]

        # Update Cluster Embedding C via Cross Attention
        Q = self.W_q(self.cluster_embeds)  # [K, d]
        K = self.W_k(h_i)  # [Batch, Channel, d]
        V = self.W_v(h_i)  # [Batch, Channel, d]

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.hidden_size).float())  # [Batch, K, Channel]
        attention_exp = torch.exp(attention_scores)
        attention_masked = attention_exp * M.transpose(1, 2)  # [Batch, K, Channel]
        attention_weights = attention_masked / attention_masked.sum(dim=-1, keepdim=True)  # Normalize

        C = torch.matmul(attention_weights, V)  # [Batch, K, d]
        self.cluster_embeds.data.copy_(C.mean(dim=0))  # Update cluster embeds with mean across batch

        # Apply TSMixer blocks
        H = self.mixer_block(h_i)  # [Batch, Channel, hidden_size]
        print(f"After mixer_block shape: {H.shape}")

        # Weight Averaging and Projection
        print("\nWeight Averaging and Projection:")
        print(f"H shape: {H.shape}")
        print(f"p_ik shape: {p_ik.shape}")
        print(f"cluster_embeds shape: {self.cluster_embeds.shape}")

        y = torch.zeros(x.size(0), self.channels, self.pred_len, device=x.device)
        for i in range(self.channels):
            theta_i = torch.sum(p_ik[:, i, :].unsqueeze(-1) * self.cluster_embeds, dim=1)  # [Batch, hidden_size]
            print(f"theta_i shape for channel {i}: {theta_i.shape}")
            print(f"H[:, i, :] shape: {H[:, i, :].shape}")
            
            y[:, i, :] = self.output_projections[i](H[:, i, :] * theta_i)
            print(f"Output for channel {i} shape: {y[:, i, :].shape}")
            # theta_i = torch.sum(p_ik[:, i, :].unsqueeze(-1) * self.cluster_embeds, dim=1)  # [Batch, hidden_size]
            # y[:, i, :] = self.output_projections[i](H[:, i, :] * theta_i)

        y = y.transpose(1, 2)  # [Batch, pred_len, Channel]
        y = self.rev_norm(y, 'denorm')
        print(f"Final output shape: {y.shape}")
        print(f"Final output: {y}")

        return y


class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, mlp_dim, dropout_factor, activation, single_layer_mixer):
        super(MlpBlockFeatures, self).__init__()
        self.batch_norm = nn.BatchNorm1d(channels)
        self.single_layer_mixer = single_layer_mixer
        if self.single_layer_mixer:
            self.linear_layer1 = nn.Linear(channels, channels)
        else:
            self.linear_layer1 = nn.Linear(channels, mlp_dim)
            self.linear_layer2 = nn.Linear(mlp_dim, channels)
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = nn.Identity()
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
        # x shape: [Batch, Channel, hidden_size]
        print("\nMlpBlockFeatures:")
        print(f"Input shape: {x.shape}")
        
        x = x.transpose(1, 2)
        print(f"After transpose shape of x: {x.shape}")

        y = self.batch_norm(x)
        print(f"After batch_norm shape: {y.shape}")
        
        y = y.transpose(1, 2)
        print(f"After transpose shape: {y.shape}")
        
        y = self.linear_layer1(y)
        print(f"After linear1 shape: {y.shape}")
        
        if self.activation_layer is not None:
            y = self.activation_layer(y)
            print(f"After activation shape: {y.shape}")
        
        if not self.single_layer_mixer:
            y = self.dropout_layer(y)
            y = self.linear_layer2(y)
            print(f"After linear2 shape: {y.shape}")
        
        y = self.dropout_layer(y)
        y = y.transpose(1, 2)
        # x = x.transpose(1, 2)
        # print(f"After transpose shape of x: {x.shape}")

        print(f"Final output shape: {y.shape}")

        return x + y


class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps"""
    def __init__(self, seq_len, dropout_factor, activation):
        super(MlpBlockTimesteps, self).__init__()
        self.batch_norm = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(seq_len, seq_len)
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = nn.Identity()
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
       # x shape: [Batch, Channel, hidden_size]
        print("\nMlpBlockTimesteps:")
        print(f"Input shape: {x.shape}")
        
        x = x.transpose(1, 2)
        print(f"After transpose shape: {x.shape}")
        
        y = self.batch_norm(x)
        print(f"After batch_norm shape: {y.shape}")

        y = y.transpose(1, 2)
        print(f"After transpose shape: {y.shape}")
        
        y = self.linear_layer(y)
        print(f"After linear shape: {y.shape}")
        
        y = y.transpose(1, 2)
        print(f"After second transpose shape: {y.shape}")
        
        y = self.activation_layer(y)
        print(f"After activation shape: {y.shape}")
        
        y = self.dropout_layer(y)
        print(f"After dropout shape: {y.shape}")
        
        return x + y

class MixerBlock(nn.Module):
    def __init__(self, channels, hidden_size, seq_len, dropout_factor, activation, single_layer_mixer, num_blocks):
        super(MixerBlock, self).__init__()
        print(f"\nMixerBlock initialization:")
        print(f"channels: {channels}, hidden_size: {hidden_size}, seq_len: {seq_len}")
        self.channels = channels
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_blocks = num_blocks

        self.timesteps_mixer = MlpBlockTimesteps(hidden_size, dropout_factor, activation)
        self.channels_mixer = MlpBlockFeatures(channels, hidden_size, dropout_factor, activation, single_layer_mixer)

    def forward(self, x):
        # x shape: [Batch, Channel, hidden_size]
        print("\nMixerBlock forward:")
        print(f"Input shape: {x.shape}")
        
        for i in range(self.num_blocks):
            print(f"\nBlock {i + 1}:")
            x = self.timesteps_mixer(x)
            print(f"After timesteps_mixer shape: {x.shape}")
            
            x = self.channels_mixer(x)
            print(f"After channels_mixer shape: {x.shape}")
        
        return x