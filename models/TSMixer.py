import torch
import torch.nn as nn
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
        batch_size, seq_len, _ = x.shape
        
        # Normalize input
        x = F.normalize(x, dim=-1)
        
        # Compute similarity matrix S
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        S = torch.exp(-torch.cdist(x, x) / (2 * x_norm**2))
        
        # Channel Embedding H via MLP
        H = self.channel_mlp(x)  # [batch_size, seq_len, hidden_dim]
        
        # Compute Clustering Probability Matrix P
        P = F.softmax(torch.matmul(H, self.cluster_embeddings.t()) / torch.norm(self.cluster_embeddings, dim=1), dim=-1)
        
        # Sample Clustering Membership Matrix M
        M = torch.bernoulli(P)
        
        # Update Cluster Embedding C via Cross Attention
        Q = self.W_Q(self.cluster_embeddings)
        K = self.W_K(H)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        C = F.normalize(torch.matmul(attention_probs, H) * M.transpose(-2, -1), dim=-1)
        
        # Update via Temporal Modules (assuming this is done in the main model)
        H_updated = H  # This will be updated by Temporal Modules in the main model
        
        # Weight Averaging and Projection
        Y = torch.zeros_like(x)
        for i in range(self.channels):
            theta_i = torch.sum(P[:, :, i].unsqueeze(-1) * self.cluster_embeddings, dim=1)
            Y[:, :, i] = self.output_projection(H_updated * theta_i.unsqueeze(1))
        
        return Y, C

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
        x = self.rev_norm(x, 'norm')
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