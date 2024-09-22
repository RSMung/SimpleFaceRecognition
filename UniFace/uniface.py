import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
import math


class UniFaceCosFace(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, l=1.0, r=1.0):
        super(UniFaceCosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # m 是余弦角度 margin，用于增加类间的可区分性
        self.m = m
        # s 是缩放因子，用于调整损失的幅度
        self.s = s
        self.my_lambda = l
        # 如果 r 增大，bias 也会相应增大，从而减小负样本损失的数值，平衡正负样本的贡献
        self.r = r
        # 公式中的 \tilde{b}
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features*r*10))

        # 权重矩阵，类别中心代理
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')

        """
        register_buffer 是 PyTorch 中的一种机制，用来注册不可训练的参数或状态。
        通过 register_buffer 注册的张量不会在优化过程中更新（即不会计算梯度），但会与模型一起保存和加载
        """
        # self.register_buffer('weight_mom', torch.zeros_like(self.weight))   # 没有用到

    def forward(self, input, label, partial_index):
        # 归一化并计算余弦相似度
        cos_theta = F.linear(
            F.normalize(input, eps=1e-5), 
            F.normalize(self.weight[partial_index], eps=1e-5)
        )

        # CosFace
        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.my_lambda

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = torch.index_select(one_hot, 1, partial_index)

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()


class UniFaceArcFace(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, l=1.0, r=1.0):
        super(UniFaceArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # m 是余弦角度 margin，用于增加类间的可区分性
        self.m = m
        # s 是缩放因子，用于调整损失的幅度
        self.s = s
        self.my_lambda = l
        # 如果 r 增大，bias 也会相应增大，从而减小负样本损失的数值，平衡正负样本的贡献
        self.r = r
        # 公式中的 \tilde{b}
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features*r*10))

        # 权重矩阵，类别中心代理
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')

        """
        register_buffer 是 PyTorch 中的一种机制，用来注册不可训练的参数或状态。
        通过 register_buffer 注册的张量不会在优化过程中更新（即不会计算梯度），但会与模型一起保存和加载
        """
        # self.register_buffer('weight_mom', torch.zeros_like(self.weight))   # 没有用到

    def forward(self, input_feats, label, partial_index):
        """
        Args:
            input (torch.tensor): [b, in_features]
            label (torch.tensor): [b]
            partial_index (torch.tensor): 形状是 p = [int(r*n_class)], 随机挑选的负样本标签序号

        Returns:
            float: 以UniFace方法为基础计算的损失值
        """
        # 归一化并计算余弦相似度
        # [b, in_feats_dim] , [p, in_feats_dim] -> [b, p]
        # print(f"input_feats:{input_feats.shape}")
        # print(f"self.weight[partial_index]:{self.weight[partial_index].shape}")
        cos_theta = F.linear(
            F.normalize(input_feats, eps=1e-5), 
            F.normalize(self.weight[partial_index], eps=1e-5)
        )

        # --------------------------- convert label to one-hot with partial_index ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # 1 表示在第二维度（即列）进行选择
        # 根据 partial_index 选择 指定的列
        # [b, n_class] -> [b, len(partial_index)]
        one_hot = torch.index_select(one_hot, 1, partial_index)

        # ArcFace
        # 计算角度值
        arc_cos = torch.acos(cos_theta)
        # 在特定位置给角度值设定margin值
        M = one_hot * self.m
        # 加上margin矩阵
        arc_cos = arc_cos + M
        # 恢复为logits
        cos_theta = torch.cos(arc_cos)

        # UniFace
        cos_m_theta_p = (self.s * cos_theta - self.bias).clamp(min=-self.s, max=self.s)
        cos_m_theta_n = cos_m_theta_p
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n)) * self.my_lambda

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()
    

def testUniFace():
    a = torch.randn(3, 5)
    label = torch.tensor([3, 1, 0])
    partial_index = torch.tensor([0, 1])
    module = UniFaceArcFace(in_features=5, out_features=4)
    b = module(a, label, partial_index)
    print(b.shape)
    print(b)


if __name__ == "__main__":
    testUniFace()