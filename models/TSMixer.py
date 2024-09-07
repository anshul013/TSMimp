import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Rev_in import RevIN

class CCM(nn.Module):
    def __init__(self, channels, hidden_dim, num_clusters):
        super(CCM, self).__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        
        # Initialize K linear layers for cluster embeddings
        self.cluster_embeddings = nn.Parameter(torch.randn(num_clusters, hidden_dim))
        
        # MLP for channel embedding
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross Attention weights
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, channels)

    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        print("Input x shape:", x.shape)
    
        # Normalize input
        x = F.normalize(x, dim=-1)
        
        # Compute similarity matrix S
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        S = torch.exp(-torch.cdist(x, x) / (2 * x_norm**2))
        print("Similarity matrix S shape:", S.shape)
        
        # Channel Embedding H via MLP
        H = self.channel_mlp(x)
        print("Channel Embedding H shape:", H.shape)
        
        # Compute Clustering Probability Matrix P
        P = F.softmax(torch.matmul(H, self.cluster_embeddings.t()) / torch.norm(self.cluster_embeddings, dim=1), dim=-1)
        print("Clustering Probability Matrix P shape:", P.shape)
        
        # Sample Clustering Membership Matrix M
        M = torch.bernoulli(P)
        print("Clustering Membership Matrix M shape:", M.shape)
        
        # Update Cluster Embedding C via Cross Attention
        Q = self.W_Q(self.cluster_embeddings)
        K = self.W_K(H)
        print("Q shape:", Q.shape)
        print("K shape:", K.shape)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        print("Attention scores shape:", attention_scores.shape)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        print("Attention probs shape:", attention_probs.shape)
        
        print("H shape for matmul:", H.shape)
        print("M shape for multiplication:", M.shape)
        
        # Compute C
        C_temp = torch.matmul(attention_probs, H)  # Shape: [32, 16, 8]
        C_temp = C_temp.transpose(1, 2)  # Shape: [32, 8, 16]
        M_transposed = M.transpose(1, 2)  # Shape: [32, 16, 512]
        C = F.normalize(torch.matmul(C_temp, M_transposed), dim=-1)  # Shape: [32, 8, 512]

        # Transpose C to match the expected output shape
        C = C.transpose(1, 2)  # Final shape: [32, 512, 8]
        print("Updated Cluster Embedding C shape:", C.shape)
        # Update via Temporal Modules (assuming this is done in the main model)
        H_updated = H + C  # Simple residual connection as temporal updatel
        print("Updated H shape after temporal update:", H_updated.shape)

        # Weight Averaging and Projection
        Y = torch.zeros_like(x)
        print("Y shape:", Y.shape)
        P_reshaped = P.transpose(1, 2)  # Shape: [32, 16, 512]
        print("P_reshaped shape:", P_reshaped.shape)
        theta = torch.sum(P_reshaped[:, :, :, None] * self.cluster_embeddings[None, :, None, :], dim=1)  # Shape: [32, 512, 8]
        print("theta shape:", theta.shape)
        # Combine H_updated and theta
        combined = H_updated * theta  # Shape: [32, 512, 8]    
        print("combined shape:", combined.shape)
        # Project to output channels
        Y = self.output_projection(combined)  # Shape: [32, 512, 7]
        print("Y shape:", Y.shape)
        # # Weight Averaging and Projection
        # Y = torch.zeros_like(x)
        # P_reshaped = P.transpose(1, 2)  # Shape: [32, 16, 512]
        # for i in range(self.channels):
        #     theta_i = torch.sum(P_reshaped[:, :, :, None] * self.cluster_embeddings[None, :, None, :], dim=1)  # Shape: [32, 512, 8]
        #     Y[:, :, i] = self.output_projection(H_updated * theta_i).squeeze(-1)
        return Y, C
        # # Weight Averaging and Projection
        # Y = torch.zeros_like(x)
        # for i in range(self.channels):
        #     theta_i = torch.sum(P[:, :, i].unsqueeze(-1) * self.cluster_embeddings, dim=1)
        #     Y[:, :, i] = self.output_projection(H_updated * theta_i.unsqueeze(1))
        
        # return Y, C

class Model(nn.Module):
    """
    Time Series Mixer model for modeling 
    channel & time steps interactions
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        #Defining the reversible instance normalization
        self.num_blocks = configs.num_blocks
        self.channels = configs.enc_in
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.hidden_size
        self.num_clusters = configs.num_clusters
        # Add ClusterChannelModule
        self.ccm = CCM(self.channels, self.hidden_dim, self.num_clusters)
        self.rev_norm = RevIN(self.channels, affine=configs.affine)
        self.mixer_block = MixerBlock(configs.enc_in, configs.hidden_size,
                                      configs.seq_len, configs.dropout,
                                        configs.activation, configs.single_layer_mixer) 

        #Individual layer for each variate(if true) otherwise, one shared linear
        self.individual_linear_layers = configs.individual
        if(self.individual_linear_layers) :
            self.output_linear_layers = nn.ModuleList()
            for _ in range(self.channels):
                self.output_linear_layers.append(nn.Linear(configs.seq_len, configs.pred_len))
        else :
            self.output_linear_layers = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # Apply CCM
        x, cluster_embedding = self.ccm(x)
        print("Shape of x before RevIN:", x.shape)
        x = self.rev_norm(x, 'norm')
        print("Shape of x after RevIN:", x.shape)
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)
        #Final linear layer applied on the transpoed mixers' output
        x = torch.swapaxes(x, 1, 2)
        # #Preparing tensor output with the correct prediction length
        y = torch.zeros([x.size(0), x.size(1), self.pred_len],dtype=x.dtype).to(x.device)
        if self.individual_linear_layers :
            for c in range(self.channels): 
                y[:, c, :] = self.output_linear_layers[c](x[:, c, :].clone())
        else :
            y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, 'denorm')
        return y, cluster_embedding

    
class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, mlp_dim, dropout_factor, activation, single_layer_mixer):
        super(MlpBlockFeatures, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(channels)
        self.single_layer_mixer = single_layer_mixer
        if self.single_layer_mixer :
            self.linear_layer1 = nn.Linear(channels, channels)
        else :
            self.linear_layer1 = nn.Linear(channels, mlp_dim)
            self.linear_layer2 = nn.Linear(mlp_dim, channels)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x) :
        y = torch.swapaxes(x, 1, 2)
        y = self.normalization_layer(y)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer1(y)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        if not(self.single_layer_mixer) :
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
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)


    def forward(self, x) :
        y = self.normalization_layer(x)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer(y)
        y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 2)
        return x + y
    
class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, channels, features_block_mlp_dims, seq_len, dropout_factor, activation, single_layer_mixer) :
        super(MixerBlock, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        #Timesteps mixing block 
        self.timesteps_mixer = MlpBlockTimesteps(seq_len, dropout_factor, activation)
        #Features mixing block 
        self.channels_mixer = MlpBlockFeatures(channels, features_block_mlp_dims, dropout_factor, activation, single_layer_mixer)
    
    def forward(self, x) :
        y = self.timesteps_mixer(x)   
        y = self.channels_mixer(y)
        return y