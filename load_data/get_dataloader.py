from torch.utils.data import DataLoader, Dataset

from load_data.cifar100 import getCIFAR100Dataset
from load_data.mnist import getMnistDataset



def build_dataloader(phase, dataset, batch_size):
    if phase == "train":
        data_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            # num_workers=2,
        )
    elif phase == "val":
        data_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
            # num_workers=2
        )
    elif phase == "test":
        data_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
            # num_workers=2
        )
    return data_dataloader


def getDataloader(
        dataset_name, phase, img_size, batch_size, 
        proportion=None, norm_type="n1"
    ):
    """
    get dataloader according to the params
    Args:
        dataset_name (_type_): _description_
        phase (_type_): train, val, test
        img_size (_type_): size of image
        batch_size (_type_): batch size
        proportion (str, optional): train:val:test. Defaults to "p3" (5:1:6).
        norm_type (str, optional): this parameter controls the mean and std for normalization when we process images.
        "n1" indicates that we use the mean and std of pre-trained vggnet. We can find more information in the offical website of torchvision. 
        "n2" indicates that we use 0.5 as the mean and std. 
        Defaults to "n1".
    Returns:
        Dataloader
    """
    # if "mnist" in dataset_name:
    #     print()
    # else:
    #     raise RuntimeError(f"dataset_name:{dataset_name} is invalid!")
    if dataset_name == "mnist":
        d = getMnistDataset(phase, img_size, norm_type)
    if dataset_name == "cifar100":
        d = getCIFAR100Dataset(phase, img_size, norm_type)
    else:
        raise RuntimeError(f"dataset_name: {dataset_name} is invalid")

    return build_dataloader(phase=phase, dataset=d, batch_size=batch_size)