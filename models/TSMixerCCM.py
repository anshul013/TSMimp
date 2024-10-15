import torch
import torch.nn as nn
from models.Rev_in import RevIN

class Model(nn.Module):
    """
    Time Series Mixer model with Channel Clustering Module (CCM)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # Defining the reversible instance normalization
        self.num_blocks = configs.num_blocks
        self.mixer_block = MixerBlock(configs.enc_in, configs.hidden_size,
                                      configs.seq_len, configs.dropout,
                                      configs.activation, configs.single_layer_mixer)
        self.channels = configs.enc_in
        self.pred_len = configs.pred_len
        self.rev_norm = RevIN(self.channels, affine=configs.affine)

        # Cluster Assigner and Embeddings
        self.num_clusters = configs.num_clusters  # K clusters
        self.hidden_size = configs.hidden_size  # Embedding size d
        self.cluster_embeds = nn.Parameter(torch.randn(self.num_clusters, self.hidden_size))  # Cluster prototypes

        # Hyperparameter σ for Similarity Matrix
        self.sigma = configs.sigma

        # Cross Attention Weights (W_q, W_k, W_v)
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size)

        # Temporal Module for updating Channel Embedding
        self.temporal_module = TemporalModule(self.hidden_size)  # Define a temporal module for embedding update

        # Linear layers for individual or shared output
        self.individual_linear_layers = configs.individual
        if self.individual_linear_layers:
            self.output_linear_layers = nn.ModuleList([nn.Linear(configs.seq_len, configs.pred_len) for _ in range(self.channels)])
        else:
            self.output_linear_layers = nn.Linear(configs.seq_len, configs.pred_len)

        # MLP for channel embeddings in the cluster assigner
        self.channel_mlp = nn.Sequential(
            nn.Linear(configs.seq_len, configs.hidden_size),
            nn.ReLU(),
            nn.Linear(configs.hidden_size, configs.hidden_size)
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.rev_norm(x, 'norm')  # Normalize input
        x = x.transpose(1, 2)  # [Batch, Channel, Input length]
        h_i = self.channel_mlp(x)  # Channel embeddings via MLP

        # Similarity calculation for clustering
        #S = torch.exp(-torch.norm(h_i.unsqueeze(1) - h_i.unsqueeze(2), dim=-1) ** 2 / (2 * 1.0 ** 2))  # Similarity matrix
        
        # Normalize channel embeddings and cluster embeddings
        h_i_normalized = h_i / (h_i.norm(dim=-1, keepdim=True) + 1e-8)  # Prevent division by zero
        c_k_normalized = self.cluster_embeds / (self.cluster_embeds.norm(dim=-1, keepdim=True) + 1e-8)  # Prevent division by zero

        # Compute clustering probability matrix P using dot product and softmax
        p_ik = torch.softmax(torch.matmul(h_i_normalized, c_k_normalized.T), dim=-1)  # [Batch, Channels, K]

        # Sample clustering membership matrix M using Bernoulli sampling
        M = torch.bernoulli(p_ik)  # Sampling clustering membership matrix

       # Update Cluster Embedding C via Cross Attention
        Q = self.W_q(self.cluster_embeds)  # [K, d]
        K = self.W_k(h_i)  # [Batch, Channels, d]
        V = self.W_v(h_i)  # [Batch, Channels, d]

        # Attention mechanism: dot product of Q and K, scaling
        attention_scores = torch.exp(torch.matmul(Q, K.transpose(-1, -2)) / (self.hidden_size ** 0.5))  # [Batch, K, Channels]
    
        # Apply membership matrix M
        M_expanded = M.permute(0, 2, 1)  # [Batch, K, Channels]
        attention_scores = attention_scores * M_expanded  # [Batch, K, Channels]
    
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [Batch, K, Channels]
    
        # Apply attention weights to V
        attention_output = torch.matmul(attention_weights, V)  # [Batch, K, d]

        # Update cluster embeddings using p_ik instead of M
        updated_cluster_embeds = torch.matmul(p_ik.transpose(1, 2), attention_output)  # [Batch, K, d]
        self.cluster_embeds.data.copy_(updated_cluster_embeds.mean(dim=0))  # Update cluster embeds

        # Update Channel Embedding via Temporal Module
        H_updated = self.temporal_module(h_i)

        # Weight Averaging
        theta = torch.einsum('bck,kd->bcd', p_ik, self.cluster_embeds)  # [Batch, Channels, d]

        # Mixing process using updated channel embeddings and averaged weights
        x = H_updated * theta  # Element-wise multiplication [Batch, Channels, d]
        x = x.transpose(1, 2)  # [Batch, Input length, Channel]
        # Apply TSMixer blocks
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)

        # Final projection for each channel
        y = torch.zeros([x.size(0), self.pred_len, self.channels], dtype=x.dtype, device=x.device)
    
        if self.individual_linear_layers:
            x = x.transpose(1, 2)  # [Batch, Channel, Input length]
            for c in range(self.channels):
                y[:, :, c] = self.output_linear_layers[c](x[:, :, c])
        else:
            y = self.output_linear_layers(x)
        
        y = y.transpose(1, 2)  # [Batch, Channel, Pred length]
        y = self.rev_norm(y, 'denorm')
        y = y.transpose(1, 2)  # [Batch, Pred length, Channel]

        return y



class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, mlp_dim, dropout_factor, activation, single_layer_mixer):
        super(MlpBlockFeatures, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(channels)
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
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
        y = torch.swapaxes(x, 1, 2)
        y = self.normalization_layer(y)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer1(y)
        if self.activation_layer is not None:
            y = self.activation_layer(y)
        if not self.single_layer_mixer:
            y = self.dropout_layer(y)
            y = self.linear_layer2(y)
        y = self.dropout_layer(y)
        return x + y


class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps with 1 layer"""
    def __init__(self, seq_len, dropout_factor, activation):
        super(MlpBlockTimesteps, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(seq_len, seq_len)
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
        y = self.normalization_layer(x)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer(y)
        y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 2)
        return x + y


class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, channels, features_block_mlp_dims, seq_len, dropout_factor, activation, single_layer_mixer):
        super(MixerBlock, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        # Timesteps mixing block 
        self.timesteps_mixer = MlpBlockTimesteps(seq_len, dropout_factor, activation)
        # Features mixing block 
        self.channels_mixer = MlpBlockFeatures(channels, features_block_mlp_dims, dropout_factor, activation, single_layer_mixer)

    def forward(self, x):
        y = self.timesteps_mixer(x)
        y = self.channels_mixer(y)
        return y


class TemporalModule(nn.Module):
    """
    Temporal Module for updating channel embeddings
    """
    def __init__(self, hidden_size):
        super(TemporalModule, self).__init__()
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.temporal_mlp(x)
    