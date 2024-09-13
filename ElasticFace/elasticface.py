
import torch.nn as nn
import torch


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class ElasticArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, mean=0.50,std=0.0125,plus=False):
        """ElasticFace: Elastic Margin Loss for Deep Face Recognition
        ElasticFace将ArcFace中的m采用从正态分布中随机采样的方式设置
        The ElasticFace+, that observes the intra-class variation
        during each training iteration and use this observation
        to assign a margin value to each sample based on its proximity
        to its class center
        Args:
            in_features (_type_): 输入特征的维度
            out_features (_type_): 输出特征的维度
            s (float, optional): 放缩因子. Defaults to 64.0.
            mean (float, optional): 用于采样m的正态分布的均值. Defaults to 0.50.
            std (float, optional): 用于采样m的正态分布的标准差. Defaults to 0.0125.
            plus (bool, optional): 是否使用ElasticFace++. Defaults to False.
        """
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.mean = mean
        self.std=std
        # 最后一层FC的权重参数
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.weight, std=0.01)
        self.plus=plus
    def forward(self, embbedings, label):
        # 将特征以及权重参数归一化处理
        embbedings = l2_norm(embbedings, axis=1)   # [b, in_feats_dim]
        weight_norm = l2_norm(self.weight, axis=0)   # [in_feats_dim, out_feats_dim]
        # 计算 cos_{\theta}
        cos_theta = torch.mm(embbedings, weight_norm)   # [b, out_feats_dim]
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        # 索引 [b], 从0   -   b-1
        index = torch.where(label != -1)[0]
        # 创建一个矩阵 [b, out_feats_dim]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        # 采样得到m值, [b, 1]
        margin = torch.normal(mean=self.mean, std=self.std, size=label[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        # 是否使用plus版本
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)

            # m_hot.scatter_(1, label[index, None], margin[idicate_cosie])

            # the corresponding position for each margin value (after sorting)
            pos = torch.stack((idicate_cosie, label[idicate_cosie]), dim=-1)
            m_hot[pos[:,0], pos[:,1]] = margin[index].squeeze()
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


def testElasticFace():
    a = torch.randn(3, 5)
    label = torch.tensor([3, 1, 0])
    module = ElasticArcFace(in_features=5, out_features=4, plus=True)
    b = module(a, label)
    print(b.shape)


if __name__ == "__main__":
    testElasticFace()