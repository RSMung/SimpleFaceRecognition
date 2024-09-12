from typing import Any, Callable, Optional, Tuple
import torchvision.datasets as vdataset
import torchvision.transforms as vtransform
import os
import torch
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np


class CIFAR100(vdataset.CIFAR100):
    def __init__(
        self,
        img_size,
        norm_type="n1",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        root = os.path.join('/home', 'RSMung', 'data', 'cifar100')
        super().__init__(root, train, transform, target_transform, download)
        if norm_type == "n1":
            # transfer learning
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif norm_type == "n2":
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            raise RuntimeError(f"norm_type:{norm_type} is invalid")
        self.normalize = vtransform.Normalize(
            mean, std
        )
        self.img_size = img_size
        self.trans_resize = vtransform.Resize(img_size)
        self.trans_to_tensor = vtransform.ToTensor()
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        # print(img.size)
        if img.size[0] != self.img_size:
            img = self.trans_resize(img)
        img = self.trans_to_tensor(img)
        img = self.normalize(img)
        return img, torch.tensor(label, dtype=torch.long)



def stratified_split(dataset, num_classes=100, train_per_class=400, val_per_class=100):
    """
    stratify
    v.(使)分层，成层
    """
    # 检查是否有idx文件
    train_indices_path = "cifar100_train_indices.npy"
    val_indices_path = "cifar100_val_indices.npy"
    if os.path.exists(train_indices_path) and os.path.exists(val_indices_path):
        # 存在保存的idx文件，直接加载使用
        train_indices = np.load(train_indices_path).tolist()
        val_indices = np.load(val_indices_path).tolist()
    else:
        # 获取idx
        train_indices = []
        val_indices = []

        # 根据类别划分数据
        class_indices = [[] for _ in range(num_classes)]

        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        # 对每个类别进行划分
        for indices in class_indices:
            train_idx, val_idx = train_test_split(indices, train_size=train_per_class, test_size=val_per_class, shuffle=False)
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
        
        # 将idx保存为文件，下次直接加载使用
        np.save(train_indices_path, np.array(train_indices))
        np.save(val_indices_path, np.array(val_indices))
    # end if else
    
    # 使用 Subset 创建训练集和验证集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def getCIFAR100Dataset(phase, img_size, norm_type):
    if phase == 'test':
        # 10000, [1, 32, 32]   -> [1, 128, 128]
        target_dataset = CIFAR100(
            img_size=img_size,
            norm_type=norm_type,
            train=False
        )
    else:
        train_val_dataset = CIFAR100(
            img_size=img_size,
            norm_type=norm_type,
            train=True
        )
    
        # train_part = 4
        # val_part = 1
        # total_per_classes = 500
        # train_per_class = int(total_per_classes * (train_part / (train_part + val_part)))  # 400
        # val_per_class = total_per_classes - train_per_class  # 100

        # num_classes=100
        # train_dataset, val_dataset = stratified_split(
        #     train_val_dataset, 
        #     num_classes=num_classes, 
        #     train_per_class=train_per_class, 
        #     val_per_class=val_per_class
        # )

        train_part = 4
        val_part = 1
        train_val_len = len(train_val_dataset)  # 50000
        train_len = int(train_val_len * (train_part / (train_part + val_part)))  # 40000
        val_len = train_val_len - train_len  # 10000
        train_dataset, val_dataset = random_split(
            train_val_dataset,
            [train_len, val_len]
        )
        
        if phase == 'train':
            target_dataset = train_dataset
        else:
            target_dataset = val_dataset
    return target_dataset
        
    
if __name__ == "__main__":
    # d = CIFAR100(img_size=128, norm_type="n1")
    # d1 = d[0][0]
    # print(d1.shape)
    # print(len(d))

    phase = "train"
    # phase = "val"
    img_size = 128
    norm_type = "n1"
    d = getCIFAR100Dataset(phase, img_size, norm_type)
    print(len(d))
    
    # # 检查每个类别拥有的数据数量
    # flag = torch.zeros((100))
    # for img, label in iter(d):
    #     flag[label] += 1
    # for i in range(100):
    #     print(f"类别 {i} 的图像有 {flag[i]} 张")