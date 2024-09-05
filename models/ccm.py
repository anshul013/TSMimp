import torch
import torch.nn as nn
import torch.nn.functional as F

class CCM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters):
        super(CCM, self).__init__()
        self.num_clusters = num_clusters
        self.cluster_embeddings = nn.Parameter(torch.randn(num_clusters, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

    def forward(self, X):
        # Step 1: Calculate channel embeddings
        channel_embeddings = self.mlp(X)  # X shape: (batch_size, num_channels, input_dim)
        
        # Step 2: Calculate clustering probabilities
        similarity = torch.einsum('bci,ki->bck', channel_embeddings, self.cluster_embeddings)  # batch, channels, clusters
        clustering_probs = F.softmax(similarity, dim=-1)

        # Step 3: Cluster-aware processing
        weighted_cluster_embeddings = torch.einsum('bck,kj->bcj', clustering_probs, self.cluster_embeddings)
        attended_output, _ = self.cross_attention(weighted_cluster_embeddings, weighted_cluster_embeddings, weighted_cluster_embeddings)
        
        return attended_output
