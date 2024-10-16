<div align="center">

## 简体中文 | [English](readme.md)

</div>

# 1 在CIFAR-100数据集上简单地复现人脸识别算法
我看完论文后发现常见的人脸识别算法论文中使用到的人脸数据集都比较大，并且需要一些预处理过程，与我单纯地想要了解算法的初衷不符合。

因此我决定在CIFAR-100数据集上复现我看的这些论文，大致了解它们的性能。

# 2 目录
- (1) ArcFace (CVPR2019)

ArcFace: Additive Angular Margin Loss for Deep Face Recognition

ArcFace 在特征与最后一层FC的权重的夹角上添加margin使得类内聚合、类间分离。

ArcFace在MNIST, CIFAR10, CIFAR100数据集上的性能均不如常规分类器方法

- (2) ElasticFace (CVPRW2022)

ElasticFace: Elastic Margin Loss for Deep Face Recognition

将ArcFace或者CosFace添加的固定margin值改为从正态分布中采样，并且cos_\theta值大的使用较小的margin值，反之使用较大的margin值。