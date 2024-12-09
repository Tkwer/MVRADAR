import torch
import torch.nn as nn

# Basic Shared Model
class BasicSharedSpecificModel(nn.Module):
    def __init__(self, feature_dim, num_views):
        super(BasicSharedSpecificModel, self).__init__()
        self.shared_layer = nn.Linear(num_views*feature_dim, feature_dim)
        self.specific_layer = nn.Linear(num_views*feature_dim, feature_dim)
        self.fusion_layer = nn.Linear(2*feature_dim, feature_dim)

    def forward(self, x_list):
        # Concatenate inputs from all views
        input = torch.cat(x_list, dim=-1)
        # Pass through shared layer
        shared_output = self.shared_layer(input)
        specific_output = self.specific_layer(input)
        output = self.fusion_layer(torch.cat([shared_output,specific_output], dim=-1))
        return output

# Attention-Enhanced Model
class AttentionEnhancedModel(nn.Module):
    def __init__(self, feature_dim, num_views):
        super(AttentionEnhancedModel, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, 1),
                nn.Softmax(dim=0)
            ) for _ in range(num_views)
        ])
        self.output_layer = nn.Linear(feature_dim * num_views, feature_dim)

    def forward(self, x_list):
        # Compute attention weights for each view
        attention_weights = [layer(x).squeeze(-1) for layer, x in zip(self.attention_layers, x_list)]
        # Apply attention weights
        attended_views = [x * w.unsqueeze(-1) for x, w in zip(x_list, attention_weights)]
        # Concatenate attended views
        concatenated = torch.cat(attended_views, dim=-1)
        # Output layer
        output = self.output_layer(concatenated)
        return output

# Adversarial Shared Model
class AdversarialSharedModel(nn.Module):
    def __init__(self, feature_dim, num_views):
        super(AdversarialSharedModel, self).__init__()
        # Shared feature extractor
        self.shared_extractor = nn.Linear(feature_dim, feature_dim)
        # Domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_views),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_list):
        # Extract shared features
        shared_features = [self.shared_extractor(x) for x in x_list]
        # Concatenate shared features
        shared_concat = torch.cat(shared_features, dim=0)
        # Domain classification
        domain_preds = self.domain_discriminator(shared_concat)
        # Compute adversarial loss externally
        return torch.mean(torch.stack(shared_features), dim=0), domain_preds

# Information Bottleneck Model
class InformationBottleneckModel(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim):
        super(InformationBottleneckModel, self).__init__()
        # Encoder
        self.encoder = nn.Linear(feature_dim, bottleneck_dim)
        # Decoder
        self.decoder = nn.Linear(bottleneck_dim, feature_dim)

    def forward(self, x_list):
        # Concatenate inputs from all views
        combined_input = torch.cat(x_list, dim=-1)
        # Encode to bottleneck
        bottleneck = self.encoder(combined_input)
        # Decode back to feature space
        reconstructed = self.decoder(bottleneck)
        return reconstructed, bottleneck

# Graph-Structured Shared Model
class GraphStructuredSharedModel(nn.Module):
    def __init__(self, feature_dim, num_views):
        super(GraphStructuredSharedModel, self).__init__()
        # Graph convolution layers for each view
        self.graph_convs = nn.ModuleList([
            GraphConvolution(feature_dim, feature_dim) for _ in range(num_views)
        ])
        # Shared aggregation layer
        self.aggregation = nn.Linear(feature_dim * num_views, feature_dim)

    def forward(self, x_list, adjacency_matrices):
        # Apply graph convolution to each view
        graph_features = [
            conv(x, adj) for conv, x, adj in zip(self.graph_convs, x_list, adjacency_matrices)
        ]
        # Concatenate graph features
        concatenated = torch.cat(graph_features, dim=-1)
        # Aggregate
        output = self.aggregation(concatenated)
        return output

# Placeholder for GraphConvolution layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output
