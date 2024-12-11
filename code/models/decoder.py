import torch
import torch.nn as nn
from models.methods.alignment import (
    MultiViewAttentionAlignment,
    MultiViewMIAlignment,
    GraphMultiViewAlignment
)
from models.methods.sharedspecific import (
    BasicSharedSpecificModel,
    AttentionEnhancedModel,
    AdversarialSharedModel,
    InformationBottleneckModel,
    GraphStructuredSharedModel
)

# 使用工厂字典映射不同的alignment和shared_specific方法
ALIGNMENT_FACTORIES = {
    'attention': lambda num_views, feature_dim: MultiViewAttentionAlignment(num_views, feature_dim),
    'mi': lambda num_views, feature_dim: MultiViewMIAlignment(num_views),
    'graph': lambda num_views, feature_dim: GraphMultiViewAlignment(num_views, feature_dim)
}

SHAREDSPECIFIC_FACTORIES = {
    'basic_shared': lambda feature_dim, num_views, bottleneck_dim=None: BasicSharedSpecificModel(feature_dim, num_views),
    'attention_enhanced': lambda feature_dim, num_views, bottleneck_dim=None: AttentionEnhancedModel(feature_dim, num_views),
    'adversarial_shared': lambda feature_dim, num_views, bottleneck_dim=None: AdversarialSharedModel(feature_dim, num_views),
    'information_bottleneck': lambda feature_dim, num_views, bottleneck_dim: InformationBottleneckModel(feature_dim * num_views, bottleneck_dim),
    'graph_structured': lambda feature_dim, num_views, bottleneck_dim=None: GraphStructuredSharedModel(feature_dim, num_views)
}

class MultiviewDecoder(nn.Module):
    """
    Base class for multi-view decoders, supports different fusion modes.
    """
    def __init__(self, feature_dim, num_views=None, mode='concat'):
        super(MultiviewDecoder, self).__init__()
        self.mode = mode
        self.num_views = num_views
        self.feature_dim = feature_dim
        # 为每个输入特征创建可学习的权重  
 
    def add_fusion(self, *x_list):
        # Dynamically sum features along the last dimension.
        # 确保 x_list 是一个包含张量的列表  
        if not x_list:  
            return None  # 如果没有输入，返回 None 或者可以抛出异常  

        # 使用 torch.stack 将所有输入张量沿着新维度堆叠  
        stacked_tensors = torch.stack(x_list, dim=0)  # 形状为 (num_views, batch_size, feature_dim)  
        
        # 对最后一个维度进行求和  
        return torch.sum(stacked_tensors, dim=0)  # 返回形状为 (batch_size, feature_dim)  

    def concat_fusion(self, *x_list):
        # Dynamically concatenate features along the last dimension.
        return torch.cat(x_list, dim=-1)

    def alignment_fusion(self, *x_list, target_domains=None):
        # Use alignment module for feature fusion
        views = list(x_list)
        return self.alignment(views)

    def shared_specific_fusion(self, *x_list, adjacency_matrices=None):
        # Different methods may require different additional inputs.
        if self.method == 'adversarial_shared':
            # Adversarial model returns both output and domain predictions
            output, domain_preds = self.shared_specific(list(x_list))
            return output
        elif self.method == 'graph_structured':
            # Graph-structured model requires adjacency matrices
            if adjacency_matrices is None:
                raise ValueError("adjacency_matrices must be provided for graph_structured method.")
            output = self.shared_specific(list(x_list), adjacency_matrices)
            return output
        elif self.method == 'information_bottleneck':
            # Information bottleneck model returns reconstructed output and bottleneck
            reconstructed, bottleneck = self.shared_specific(list(x_list))
            return reconstructed
        else:
            # Other methods return the fused output directly
            output = self.shared_specific(list(x_list))
            return output

    def forward(self, *x_list, **kwargs):
        if self.num_views is not None and len(x_list) != self.num_views:
            raise ValueError(f"Expected {self.num_views} inputs, got {len(x_list)}")

        if self.mode == 'add':
            return self.add_fusion(*x_list)
        elif self.mode == 'concat':
            return self.concat_fusion(*x_list)
        elif self.mode == 'alignment':
            return self.alignment_fusion(*x_list, **kwargs)
        elif self.mode == 'shared_specific':
            return self.shared_specific_fusion(*x_list, **kwargs)
        else:
            raise ValueError("Invalid fusion mode")

# add 是concat的特殊形式
class ConcatDecoder(MultiviewDecoder):  
    """  
    Simply concatenates input features from multiple views.  
    """  
    def __init__(self, feature_dim, num_views=None, mode='concat'):  
        super().__init__(feature_dim=feature_dim, num_views=num_views, mode=mode)  
        # 计算输入特征的大小  
        if mode=='concat':
            input_size = feature_dim * num_views  # 假设每个视图的特征维度均为 feature_dim  
        elif mode=='add':
            input_size = feature_dim  # 假设每个视图的特征维度均为 feature_dim 

        # 添加线性层，输入大小为 input_size，输出大小为 feature_dim  
        self.linear = nn.Linear(input_size, feature_dim)  
        
    def forward(self, *x_list, **kwargs):  
        # 调用父类的 forward 方法  
        fused_features = super().forward(*x_list, **kwargs)  
        
        # 通过线性层输出  
        output = self.linear(fused_features)  
        
        return output, None 


class AlignmentDecoder(MultiviewDecoder):  
    """  
    Aligns multiple views using different alignment strategies.  
    """  
    def __init__(self, feature_dim, num_views=None, method='attention'):  
        super().__init__(feature_dim=feature_dim, num_views=num_views, mode='alignment')  
        if method not in ALIGNMENT_FACTORIES:  
            raise ValueError(f"Unknown alignment type: {method}")  
        self.method = method  
        self.alignment = ALIGNMENT_FACTORIES[method](num_views,feature_dim)  

    def forward(self, *x_list, **kwargs):  
        # 直接调用父类的 forward 方法  
        return super().forward(*x_list, **kwargs)  


class SharedSpecificDecoder(MultiviewDecoder):  
    """  
    Handles shared and specific feature extraction/fusion.  
    """  
    def __init__(self, feature_dim, num_views=None, method='basic_shared', bottleneck_dim=None):  
        super().__init__(feature_dim=feature_dim, num_views=num_views, mode='shared_specific')  
        if method not in SHAREDSPECIFIC_FACTORIES:  
            raise ValueError(f"Unknown shared_specific method: {method}")  
        if method == 'information_bottleneck' and bottleneck_dim is None:  
            raise ValueError("bottleneck_dim must be specified for information bottleneck method.")  
        self.method = method  
        self.shared_specific = SHAREDSPECIFIC_FACTORIES[method](feature_dim,num_views,bottleneck_dim)  

    def forward(self, *x_list, **kwargs):  
        # 直接调用父类的 forward 方法  
        return super().forward(*x_list, **kwargs)