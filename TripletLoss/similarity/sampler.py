import random
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler


def create_groups(groups, k):
    """Bins sample indices with respect to groups, remove bins with less than k samples

    Args:
        groups (list[int]): where ith index stores ith sample's group id

    Returns:
        defaultdict[list]: Bins of sample indices, binned by group_idx
    """
    group_samples = defaultdict(list)

    # print(f"groups: {groups}")
    # print(f"groups: {type(groups)}")
    # print(f"groups: {len(groups)}")

    # 样本标签, 类别标签
    # 将同一个类别的样本的索引放到一起
    for sample_idx, group_idx in enumerate(groups):
        group_samples[group_idx].append(sample_idx)

    # 检查哪些类别的样本数量少于k个
    keys_to_remove = []
    for key in group_samples:
        # print(f"key-nums: {key}-{len(group_samples[key])}")
        if len(group_samples[key]) < k:
            keys_to_remove.append(key)
            continue

    # 将这些类别数据剔除
    for key in keys_to_remove:
        group_samples.pop(key)

    return group_samples


class PKSampler(Sampler):
    """
    Randomly samples from a dataset  while ensuring that each batch (of size p * k)
    includes samples from exactly p labels, with k samples for each label.

    Args:
        groups (list[int]): List where the ith entry is the group_id/label of the ith sample in the dataset.
        p (int): Number of labels/groups to be sampled from in a batch
        k (int): Number of samples for each label/group in a batch
    """

    def __init__(self, groups, p, k):
        self.p = p   # 每个batch要求有多少个类别
        self.k = k   # 每个类别要求多少个样本
        # 一个字典，key是类别标签，value是存放了该类别所有样本的索引的list
        self.groups = create_groups(groups, self.k)
        # print(f"self.groups:{len(self.groups)}")
        # print(f"p:{p}")

        # Ensures there are enough classes to sample from
        if len(self.groups) < p:
            raise ValueError("There are not enough classes to sample from")

    def __iter__(self):
        # Shuffle samples within groups
        for key in self.groups:
            random.shuffle(self.groups[key])

        # 记录每组剩余的样本数量
        # Keep track of the number of samples left for each group
        group_samples_remaining = {}
        for key in self.groups:
            group_samples_remaining[key] = len(self.groups[key])

        while len(group_samples_remaining) > self.p:
            # 字典中的所有键转换为一个列表，并赋值给 group_ids
            # Select p groups at random from valid/remaining groups
            group_ids = list(group_samples_remaining.keys())
            # 创建一个长度为 len(group_ids) 的一维张量，所有元素都为 1。这表示每个组 ID 的初始权重相等
            # p代表抽样的数量
            # 根据给定的权重（这里是全为 1 的张量）随机选择 self.p 个索引
            selected_group_idxs = torch.multinomial(torch.ones(len(group_ids)), self.p).tolist()
            # 遍历被选中的类别
            for i in selected_group_idxs:
                # 得到第 i 个类别的类别标签
                group_id = group_ids[i]
                # 得到该类别的数据
                group = self.groups[group_id]
                for _ in range(self.k):
                    # No need to pick samples at random since group samples are shuffled
                    # 样本已经被打乱，因此可以简单地从后面开始提取
                    sample_idx = len(group) - group_samples_remaining[group_id]
                    # 暂停函数的执行，并返回当前的值
                    # 当生成器函数再次被调用时，它会从上一次执行 yield 的地方继续执行，而不是从头开始
                    yield group[sample_idx]
                    # 剩余样本数量减去1
                    group_samples_remaining[group_id] -= 1

                # 如果当前类别剩余的样本数量不够k个则删除该类别
                # Don't sample from group if it has less than k samples remaining
                if group_samples_remaining[group_id] < self.k:
                    group_samples_remaining.pop(group_id)