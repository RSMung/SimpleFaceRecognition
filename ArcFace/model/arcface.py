import torch
import torch.nn as nn
import torch.nn.functional as F



class ArcFaceLoss(nn.Module):
    def __init__(self, feat_dim:int, n_class:int, scale=10, margin=0.5):
        """
        Args:
            feat_dim (int): 特征维度
            n_class (int): 类别数量
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.n_class = n_class
        self.scale = scale   # 放缩系数
        self.margin = torch.tensor(margin)   # margin值
        # 最后一个全连接层的权重参数
        self.weight = nn.Parameter(torch.rand(feat_dim, n_class), requires_grad=True)
        nn.init.xavier_uniform_(self.weight) # Xavier 初始化 FC 权重
    
    def forward(self, feats:torch.Tensor, labels:torch.Tensor):
        # print(feats.shape)
        # print(self.weight.shape)
        # 归一化特征向量以及权重参数，然后计算它们的矩阵乘法
        cos_theta = torch.matmul(F.normalize(feats), F.normalize(self.weight))
        # 防止数值问题
        cos_theta = cos_theta.clip(-1+1e-7, 1-1e-7)
        
        # 计算角度值
        arc_cos = torch.acos(cos_theta)
        # 在特定位置给角度值设定margin值
        M = F.one_hot(labels, num_classes = self.n_class) * self.margin
        # 加上margin矩阵
        arc_cos = arc_cos + M
        
        # 恢复为logits
        cos_theta_2 = torch.cos(arc_cos)
        # 放缩
        logits = cos_theta_2 * self.scale
        return logits