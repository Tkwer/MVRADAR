import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    

# MultiView 对抗性多视角对齐  
class DomainAdversarialNetwork(nn.Module):  
    def __init__(self, feature_dim, num_domains):  
        super().__init__()  
        # 域判别器  
        self.domain_classifier = nn.Sequential(  
            nn.Linear(feature_dim, feature_dim // 2),  
            nn.ReLU(),  
            nn.Linear(feature_dim // 2, num_domains),  
            nn.Softmax(dim=-1)  
        )  


    def forward(self, features, alpha):  

          # 通过反向梯度层
        reverse_features = ReverseLayerF.apply(features, alpha)

        # 域分类
        domain_logits = self.domain_classifier(reverse_features)
        return domain_logits