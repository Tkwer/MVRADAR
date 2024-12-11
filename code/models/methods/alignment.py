import torch  
import torch.nn as nn  
from torch.nn import MultiheadAttention  


# 图卷积的简单实现（用来替代原代码中未定义的 GraphConvolution）  
class GraphConvolution(nn.Module):  
    def __init__(self, in_features, out_features):  
        super(GraphConvolution, self).__init__()  
        self.linear = nn.Linear(in_features, out_features)  

    def forward(self, features, adjacency_matrix):  
        """  
        朴素的图卷积实现：A_hat * X * W  
        :param features: 节点特征矩阵 (N, F_in)  
        :param adjacency_matrix: 邻接矩阵 (N, N)  
        :return: 输出特征矩阵 (N, F_out)  
        """  
        support = self.linear(features)  # X * W  
        out = torch.matmul(adjacency_matrix, support)  # A * X * W  
        return out  


# MultiView 线性注意力对齐  
class MultiViewAttentionAlignment(nn.Module):  
    def __init__(self, num_views, feature_dim):  
        super().__init__()  
        # 每个视角的注意力权重模块  
        self.view_attention = nn.ModuleList([  
            nn.Sequential(  
                nn.Linear(feature_dim, 1),  
                nn.Softmax(dim=-1)  
            ) for _ in range(num_views)  
        ])  

        # 融合层  
        self.fusion = nn.Linear(feature_dim * num_views, feature_dim)  

    def forward(self, views):  
        """  
        views: 一个包含多个视角特征的列表  
        每个视角形状为 (batch_size, feature_dim)  
        """  
        aligned_views = []  
        view_weights =[]
        for i, view in enumerate(views):  
            # 每个视角的注意力加权计算  
            view_weight = self.view_attention[i](view)  
            aligned_view = view * view_weight  # (batch_size, feature_dim)  
            aligned_views.append(aligned_view)  
            view_weights.append(view_weight)

        # 跨视角融合  
        view_weights = torch.cat(view_weights, dim=-1)
        cross_view = torch.cat(aligned_views, dim=-1)  # (batch_size, feature_dim * num_views)  
        alignment = self.fusion(cross_view)  # (batch_size, feature_dim)  

        return alignment, view_weights 


# MultiView 互信息最大化对齐  在提取公共表征可以使用 互信息最大化对齐（构建对应的loss）
class MultiViewMIAlignment(nn.Module):  
    def __init__(self, num_views):  
        super().__init__()  


    def calculate_mi(self, x, y):  
        """  
        一个简单的互信息计算占位符函数  
        实际实现中需要更复杂的统计方法，例如基于直方图估计或神经网络估计  
        """  
        # 假设这里简单返回 x 和 y 的内积作为互信息  
        return torch.sum(x * y) / (x.size(0) * x.size(1))  

    def compute_mutual_information(self, views):  
        """  
        计算多个视角之间的互信息矩阵  
        """  
        num_views = len(views)  
        mi_matrix = torch.zeros(num_views, num_views)  # 用来存储互信息值  
        for i in range(num_views):  
            for j in range(i + 1, num_views):  
                mi_matrix[i, j] = self.calculate_mi(views[i], views[j])  
                mi_matrix[j, i] = mi_matrix[i, j]  # 对称矩阵  
        return mi_matrix  

    def forward(self, views):  
        """  
        views: 一个包含多个视角特征的列表  
        每个视角形状为 (batch_size, feature_dim)  
        """  
        # 计算视角的互信息矩阵  
        mi_matrix = self.compute_mutual_information(views)  

        # 根据互信息加权调整每个视角  
        aligned_views = []  
        view_weights =[]
        for i, view in enumerate(views):  
            weight = torch.sum(mi_matrix[i])
            aligned_views.append(view * weight) 
            view_weights.append(weight) 
        view_weights = torch.cat(view_weights, dim=-1)
        # 合并所有对齐后的视角  
        return torch.mean(torch.stack(aligned_views) , dim=0) ,view_weights 


# MultiView 图网络多视角对齐  
class GraphMultiViewAlignment(nn.Module):  
    def __init__(self, num_views, feature_dim):  
        super().__init__()  
        # 图卷积层  
        self.graph_conv = GraphConvolution(feature_dim, feature_dim)  

        # 多头注意力  
        self.view_attention = MultiheadAttention(  
            embed_dim=feature_dim,  
            num_heads=num_views,  
            batch_first=True  
        )  

    def compute_view_similarity(self, views):  
        """  
        计算视角之间的相似性矩阵  
        """  
        num_views = len(views)  
        similarity_matrix = torch.zeros(num_views, num_views)  
        for i in range(num_views):  
            for j in range(num_views):  
                similarity_matrix[i, j] = torch.cosine_similarity(  
                    views[i].mean(dim=0, keepdim=True),  
                    views[j].mean(dim=0, keepdim=True)  
                )  
        return similarity_matrix  

    def construct_view_graph(self, views):  
        """  
        根据视角相似性矩阵构建视角图  
        """  
        similarity_matrix = self.compute_view_similarity(views)  
        adjacency_matrix = (similarity_matrix > 0.5).float()  # 设置一个简单的阈值，作为邻接矩阵  
        return adjacency_matrix  

    def forward(self, views):  
        """  
        views: 一个包含多个视角特征的列表  
        每个视角形状为 (batch_size, feature_dim)  
        """  
        view_graph = self.construct_view_graph(views)  # 构建图邻接矩阵  

        # 图卷积处理每个视角  
        graph_views = [  
            self.graph_conv(view, view_graph)  
            for view in views  
        ]  

        # 使用多头注意力聚合视角  
        stacked_views = torch.stack(graph_views, dim=1)  # (batch_size, num_views, feature_dim)  
        aligned_view, _ = self.view_attention(stacked_views, stacked_views, stacked_views)  

        return aligned_view.mean(dim=1)  # 返回最终的对齐视角  


#