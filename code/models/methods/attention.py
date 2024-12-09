import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewSEAttention(nn.Module):  
    def __init__(self, num_views, feature_dim):  
        super().__init__()  
        # 全局平均池化 + 权重生成  
        self.fc_layers = nn.Sequential(  
                nn.Linear(num_views, num_views // 2),  
                nn.ReLU(inplace=True),  
                nn.Linear(num_views // 2, num_views),  
                nn.Sigmoid()  
            )

    def forward(self, views):  
        views_tensor = torch.stack(views, dim=1) #形状为 (batch_size, num_views, feature_dim)
        view_weights = F.adaptive_avg_pool1d(views_tensor, 1).squeeze() #形状为 (batch_size, num_views)
        view_weights = self.fc_layers(view_weights)    #形状为 (batch_size, num_views)
        aligned_views = views * view_weights.unsqueeze(-1)  
        # 简单拼s接或求平均  
        return aligned_views
    
class MultiViewECAAttention(nn.Module):
    def __init__(self, num_views, feature_dim, kernel_size=3):
        super().__init__()
        
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        
        # ECA模块：用于生成每个视角的注意力权重
        self.conv = nn.Conv1d(in_channels=num_views, out_channels=1, kernel_size=self.kernel_size, padding=self.kernel_size // 2)

    def forward(self, views):
        """
        views: 长度为 num_views 的列表，每个视角的形状为 (batch_size, feature_dim)
        """
        views_tensor = torch.stack(views, dim=1)  # 形状为 (batch_size, num_views, feature_dim)
        
        # 将特征维度移到最后一维，并使用卷积生成注意力权重
        views_tensor = views_tensor.permute(0, 2, 1)  # (batch_size, feature_dim, num_views)
        
        # 使用1D卷积生成视角注意力权重
        view_weights = self.conv(views_tensor)  # (batch_size, 1, num_views)
        view_weights = view_weights.squeeze(1)  # (batch_size, num_views)
        view_weights = F.sigmoid(view_weights)  # (batch_size, num_views)

        # 直接使用 view_weights 来加权每个视角
        aligned_views = views * view_weights.unsqueeze(-1)  # (batch_size, num_views, feature_dim)

        return aligned_views
    
class AdaptiveMultiViewAttention(nn.Module):
    def __init__(self, num_views, feature_dim):
        super().__init__()
        # 交互式注意力模块
        self.cross_view_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4
        )
        
        # 权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_views)
        )

        # 线性变换层，用于生成 Q, K, V
        self.query_linear = nn.Linear(feature_dim, feature_dim)
        self.key_linear = nn.Linear(feature_dim, feature_dim)
        self.value_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, views):
        # 转换为序列形式 (num_views, batch_size, feature_dim)
        views_tensor = torch.stack(views)  # (num_views, batch_size, feature_dim)
        
        # 生成 Q, K, V
        Q = self.query_linear(views_tensor)  # (num_views, batch_size, feature_dim)
        K = self.key_linear(views_tensor)    # (num_views, batch_size, feature_dim)
        V = self.value_linear(views_tensor)  # (num_views, batch_size, feature_dim)

        # 自注意力交互
        attn_output, _ = self.cross_view_attention(Q, K, V)
        
        # 权重生成
        view_weights = self.weight_generator(attn_output.mean(dim=0))  # (batch_size, num_views)
        view_weights = F.softmax(view_weights, dim=-1)  # (batch_size, num_views)
        
        # 加权求和
        aligned_views = []
        for i in range(len(views)):
            aligned_view = views[i] * view_weights[:, i].unsqueeze(-1)  # (batch_size, feature_dim)
            aligned_views.append(aligned_view)
        
        # 返回加权平均的结果
        return torch.stack(aligned_views).mean(dim=0)  # (batch_size, feature_dim)