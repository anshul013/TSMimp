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
            nn.Linear(self.channels, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.rev_norm(x, 'norm')  # Normalize input
        h_i = self.channel_mlp(x)  # Channel embeddings via MLP

        # Similarity calculation for clustering
        #S = torch.exp(-torch.norm(h_i.unsqueeze(1) - h_i.unsqueeze(2), dim=-1) ** 2 / (2 * 1.0 ** 2))  # Similarity matrix
        # Normalize channel embeddings and cluster embeddings
        h_i_normalized = h_i / (h_i.norm(dim=-1, keepdim=True) + 1e-8)  # Prevent division by zero
        c_k_normalized = self.cluster_embeds / (self.cluster_embeds.norm(dim=-1, keepdim=True) + 1e-8)  # Prevent division by zero

        # Compute similarity scores using normalized embeddings
        similarity_scores = torch.matmul(h_i_normalized, c_k_normalized.T)  # [Batch, Channels, K]
    
        # Compute clustering probability matrix P using softmax
        p_ik = torch.softmax(similarity_scores, dim=-1)  # Normalize to get probabilities
        
        M = torch.bernoulli(p_ik)  # Sampling clustering membership matrix

        # Update Cluster Embedding C via Cross Attention
        Q = self.W_q(self.cluster_embeds)  # [K, d]
        K = self.W_k(h_i)  # [Batch, Channels, d]
        V = self.W_v(h_i)  # [Batch, Channels, d]

        # Attention mechanism: dot product of Q and K, scaling, and applying to V
        attention_weights = torch.softmax(torch.exp((Q @ K.transpose(-1, -2)) / (self.hidden_size ** 0.5)) @ M.T, dim=-1)
        attention_output = torch.einsum('bcd,kc->bkd', attention_weights, V)  # Apply attention weights to V

        # Update cluster embeddings using weighted average
        updated_cluster_embeds = (attention_output.permute(0, 2, 1) * p_ik.unsqueeze(-1)).sum(dim=1)  # [Batch, d]
        self.cluster_embeds.data.copy_(updated_cluster_embeds.mean(dim=0))  # Update cluster embeds

        # Update Channel Embedding via Temporal Module
        updated_channel_embeds = self.temporal_module(h_i)  # Use MlpBlockTimesteps or the custom Temporal Module

        # Mixing process using updated channel embeddings
        x = updated_channel_embeds  # Use updated channel embeddings for mixing
        x = self.rev_norm(x, 'norm')  # Normalize input X: [Batch, Input length, Channel]
    
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)

        # Final linear layer applied on the transposed mixer's output
        x = torch.swapaxes(x, 1, 2)

        # Prepare tensor output with the correct prediction length
        y = torch.zeros([x.size(0), x.size(1), self.pred_len], dtype=x.dtype).to(x.device)
    
        if self.individual_linear_layers:
            for c in range(self.channels):
                y[:, c, :] = self.output_linear_layers[c](x[:, c, :].clone())
        else:
            y = self.output_linear_layers(x.clone())

        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, 'denorm')

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
