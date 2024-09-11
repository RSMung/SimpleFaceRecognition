from typing import Any, Callable, Optional, Tuple
import torchvision.datasets as vdataset
import torchvision.transforms as vtransform
import os
import torch
from torch.utils.data import Dataset, random_split

class MNIST(vdataset.MNIST):
    def __init__(
        self,
        img_size,
        norm_type,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        root = os.path.join('/home', 'RSMung', 'data', 'mnist')
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
        # self.normalize = vtransform.Normalize((0.5), (0.5))
        self.trans_resize = vtransform.Resize(img_size)
        self.trans_to_tensor = vtransform.ToTensor()
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        # print(img.size)
        if img.size[0] != self.img_size:
            img = self.trans_resize(img)
        img = self.trans_to_tensor(img)
        img = img.repeat(3, 1, 1)   # 转3通道
        img = self.normalize(img)
        return img, torch.tensor(label, dtype=torch.long)



def getMnistDataset(phase, img_size, norm_type):
    if phase == 'test':
        # 10000, [1, 28, 28]
        target_dataset = MNIST(
            img_size=img_size,
            norm_type=norm_type,
            train=False
        )
    else:
        train_val_dataset = MNIST(
            img_size=img_size,
            norm_type=norm_type,
            train=True
        )
    
        train_part = 5
        val_part = 1
        train_val_len = len(train_val_dataset)  # 60000
        train_len = int(train_val_len * (train_part / (train_part + val_part)))  # 50000
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
    d = MNIST(img_size=224)
    d1 = d[0][0]
    print(d1.shape)
    print(len(d))
    # import torchvision.utils as vutils
    # vutils.save_image(d1, "mnist_sample.png")