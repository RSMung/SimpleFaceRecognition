import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

# 固定随机数种子
# set the random seed
def prepareEnv(seed = 1):
    import torch.backends.cudnn as cudnn

    # controls whether cuDNN is enabled. cudnn could accelerate the training procedure
    # cudnn.enabled = False
    cudnn.enabled = True
    # 使得每次返回的卷积算法是一样的
    # if True, causes cuDNN to only use deterministic convolution algorithms
    cudnn.deterministic = True
    # 如果网络的输入数据维度或类型上变化不大, 可以增加运行效率
    # 自动寻找最适合当前的高效算法,优化运行效率
    # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
    # cudnn.benchmark = False
    cudnn.benchmark = True
    

    """
    在需要生成随机数据的实验中，每次实验都需要生成数据。
    设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
    """
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数
    torch.cuda.manual_seed_all(seed)  # 给所有GPU设置
    np.random.seed(seed)
    random.seed(seed)



# class ParamsParent:
#     """
#     各个参数类的父类
#     """
#     def __repr__(self):
#         # 直接打印这个类时会调用这个函数, 打印返回的输出的字符串
#         str_result = f"---{self.__class__.__name__}---\n"
#         # 剔除带__的属性
#         # dir(self.__class__)会返回属性的有序列表
#         # self.__dir__()返回属性列表, 与前者的区别是不会排序
#         for attr in self.__dir__():
#             if not attr.startswith('__'):
#                 str_result += "{}: {}\n".format(attr, self.__getattribute__(attr))
#         str_result += "------------------\n"
#         return str_result
    
def get_rootdir_path():
    # 当前文件的路径
    current_path = os.path.abspath(__file__)

    # 当前文件所在的目录
    root_dir = os.path.dirname(current_path)
    return root_dir


@torch.no_grad()
def identification_procedure(
    train_feats, train_labels, 
    query_feats, query_labels
):
    """
    credits: https://github.com/weixu000/DSH-pytorch/blob/906399c3b92cf8222bca838c2b2e0e784e0408fa/utils.py#L58
    查询特征与注册集中的特征匹配, 看和哪个最接近以确定预测标签, 然后基于此计算精度acc
    Args:
        train_feats (_type_): 注册集特征   [train_num, feats_dim]
        train_labels (_type_): 注册集标签   [train_num]
        query_feats (_type_): 查询集特征   [query_num, feats_dim]
        query_labels (_type_): 查询集标签   [query_num]

    Returns:
        float: acc
    """
    correct_num = 0
    query_samples_num = query_feats.size(0)

    # 将数据移动到 GPU（如果可用）
    train_feats, train_labels = train_feats.cuda(), train_labels.cuda()
    query_feats, query_labels = query_feats.cuda(), query_labels.cuda()

    # print(f"query_feats shape:{query_feats.shape}")
    # print(f"train_feats shape:{train_feats.shape}")

    # 计算余弦相似度
    query_feats = F.normalize(query_feats)   # 归一化, 将某一个维度除以那个维度对应的范数(默认是2范数, dim=1)
    train_feats = F.normalize(train_feats)
    cosine_similarity = torch.matmul(query_feats, train_feats.transpose(0,1))   # [query_num, train_num]
    # print(f"cosine_similarity: {cosine_similarity.shape}")

    _, predicted_idx = torch.max(cosine_similarity, dim=1)   # [query_num]

    predicted_labels = train_labels[predicted_idx]

    # 将预测标签与query的真实标签比较
    correct_mask = (query_labels == predicted_labels)
    # print(f"correct_mask: {correct_mask.shape}")

    # 如果检索成功，则计数器加一
    correct_num = correct_mask.sum().item()

    # 计算精度
    # print(f"correct_num: {correct_num}")
    # print(f"num_samples: {num_samples}")
    acc = correct_num / query_samples_num
    # print(f"acc: {acc}")

    return acc


@torch.no_grad()
def verification_procedure(
    train_feats, train_labels, 
    query_feats, query_labels,
    model_name=None, dataset_name=None,
    save_csv=False
):
    # query_samples_num = query_feats.size(0)

    # 将数据移动到 GPU（如果可用）
    train_feats, train_labels = train_feats.cuda(), train_labels.cuda()
    query_feats, query_labels = query_feats.cuda(), query_labels.cuda()

    # print(f"train_labels shape: {train_labels.shape}")
    # print(f"query_labels shape: {query_labels.shape}")
    
    # 计算余弦相似度
    query_feats = F.normalize(query_feats)   # 归一化, 将某一个维度除以那个维度对应的范数(默认是2范数, dim=1)
    train_feats = F.normalize(train_feats)
    cosine_similarity = (query_feats @ train_feats.transpose(0,1)).to(torch.float16)   # [query_num, train_num]
    # cosine_distance = 1 - cosine_similarity
    del query_feats, train_feats

    # 构造pairs的真假标签   [query_num, train_num]
    genuine_or_imposter_labels = torch.eq(query_labels[:, None], train_labels[None, :]).to(torch.int8)
    # print(f"genuine_or_imposter_labels: {genuine_or_imposter_labels.shape}")
    del query_labels, train_labels

    # genuine_idxs = torch.where(genuine_or_imposter_labels == 1)
    # genuine_cosine_distance = cosine_similarity[genuine_idxs]

    # imposter_idxs = torch.where(genuine_or_imposter_labels == 0)
    # imposter_cosine_distance = cosine_similarity[imposter_idxs]

    cosine_similarity = cosine_similarity.cpu()
    genuine_or_imposter_labels = genuine_or_imposter_labels.cpu()

    
    #region sklearn roc_curve
    # flatten and convert to numpy matrix
    cosine_similarity = torch.flatten(cosine_similarity).numpy()
    genuine_or_imposter_labels = torch.flatten(genuine_or_imposter_labels).numpy()
    # utilizing the sklearn function
    from sklearn.metrics import roc_curve
    all_far_data, tar, thresholds = roc_curve(genuine_or_imposter_labels, cosine_similarity)
    all_frr_data = 1 - tar
    import numpy as np
    eer = all_far_data[np.nanargmin(np.absolute((all_far_data - all_frr_data)))]
    #endregion


    if save_csv:
        assert model_name is not None and dataset_name is not None
        # 保存 det 数据到csv文件
        import os
        import pandas as pd
        # tar_data, far_data = get_roc_data_cdtrans(y_true, y_score)
        current_path = os.path.abspath(__file__)   # 本文件的目录
        root_dir = os.path.dirname(current_path)   # 本文件的父目录
        # tar
        pdOperator_far = pd.DataFrame(
            data={"FAR":all_far_data, "FRR":all_frr_data}
        )
        pdOperator_far.to_csv(
            os.path.join(
                root_dir,
                model_name+"_far_frr_" + dataset_name+ ".csv"
            ),
            index=False, 
            # header=False
        )

    return eer