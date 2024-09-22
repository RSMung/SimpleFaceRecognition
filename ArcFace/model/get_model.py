
import torch
import torch.nn as nn
import os

from ArcFace.model.arcface import ArcFaceLoss
from ElasticFace.elasticface import ElasticArcFace
from UniFace.uniface import UniFaceArcFace
from global_utils import get_rootdir_path
from ArcFace.model.resnet import Resnet18, Resnet18_softmax

loss_func_module_ckp_name = {
    "arcface":"arcface_module",
    "ElasticFacePlus-Arc":"elasticfaceplusarc_module",
    "UniFaceArcFace":"UniFaceArcFace"
}

def load_resnet18_arcface_ckp(cls_model:nn.Module, arcface_loss_func, 
                              dataset_name, backbone_type, loss_fuc_type, ckp_time_stamp):
    if loss_fuc_type == "arcface":
        parent_dir = "ArcFace"
    elif loss_fuc_type == "ElasticFacePlus-Arc":
        parent_dir = "ElasticFace"
    elif loss_fuc_type == "UniFaceArcFace":
        parent_dir = "UniFace"
    else:
        raise RuntimeError(f"loss_fuc_type:{loss_fuc_type} is invalid!")
    
    model_name = backbone_type + "_" + loss_fuc_type

    ckp_path = os.path.join(
        get_rootdir_path(), parent_dir, "ckp", ckp_time_stamp,
        dataset_name + "_" + model_name + "_" + ckp_time_stamp
    )
    print(f"ckp_path:{ckp_path} 已被加载！")

    # load archive
    archive = torch.load(ckp_path)
    
    cls_model.load_state_dict(archive["cls_model"])
    arcface_loss_func.load_state_dict(
        archive[
            loss_func_module_ckp_name[loss_fuc_type]
        ]
    )
    return cls_model, arcface_loss_func

#region old code
# def load_resnet18_arcface_ckp(cls_model:nn.Module, arcface_loss_func, 
#                               dataset_name, model_name, ckp_time_stamp, loss_fuc_type):
#     ckp_path = os.path.join(
#         get_rootdir_path(), "ArcFace", "ckp", ckp_time_stamp,
#         dataset_name + "_" + model_name + "_" + ckp_time_stamp
#     )
#     print(f"ckp_path:{ckp_path} 已被加载！")
#     # load archive
#     archive = torch.load(ckp_path)
#     cls_model.load_state_dict(archive["cls_model"])
#     arcface_loss_func.load_state_dict(
#         archive[
#             loss_func_module_ckp_name[loss_fuc_type]
#         ]
#     )
#     return cls_model, arcface_loss_func

# def load_resnet18_elasticfaceplusarc_ckp(cls_model:nn.Module, elasticfaceplusarc_loss_func, 
#                                          dataset_name, model_name, ckp_time_stamp, loss_fuc_type):
#     ckp_path = os.path.join(
#         get_rootdir_path(), "ElasticFace", "ckp", ckp_time_stamp,
#         dataset_name + "_" + model_name + "_" + ckp_time_stamp
#     )
#     print(f"ckp_path:{ckp_path} 已被加载！")
#     # load archive
#     archive = torch.load(ckp_path)
#     cls_model.load_state_dict(archive["cls_model"])
#     elasticfaceplusarc_loss_func.load_state_dict(
#         archive[
#             loss_func_module_ckp_name[loss_fuc_type]
#         ]
#     )
#     return cls_model, elasticfaceplusarc_loss_func

# def load_resnet18_unifacearc_ckp(cls_model:nn.Module, unifacearc_loss_func, 
#                                  dataset_name, model_name, ckp_time_stamp, loss_fuc_type):
#     ckp_path = os.path.join(
#         get_rootdir_path(), "UniFace", "ckp", ckp_time_stamp,
#         dataset_name + "_" + model_name + "_" + ckp_time_stamp
#     )
#     print(f"ckp_path:{ckp_path} 已被加载！")
#     # load archive
#     archive = torch.load(ckp_path)
#     cls_model.load_state_dict(archive["cls_model"])
#     unifacearc_loss_func.load_state_dict(
#         archive[
#             loss_func_module_ckp_name[loss_fuc_type]
#         ]
#     )
#     return cls_model, unifacearc_loss_func
#endregion


def load_resnet18_softmax_ckp(cls_model:nn.Module, dataset_name, 
                              model_name, ckp_time_stamp):
    ckp_path = os.path.join(
        get_rootdir_path(), "ArcFace", "ckp", ckp_time_stamp,
        dataset_name + "_" + model_name + "_" + ckp_time_stamp
    )
    print(f"ckp_path:{ckp_path} 已被加载！")
    # load archive
    cls_model.load_state_dict(torch.load(ckp_path))
    return cls_model



def defineModel(
        dataset_name, backbone_type, loss_fuc_type,
        n_class, 
        ckp_time_stamp=None, 
        # img_size=128
    )->nn.Module:
    """
    define the model structure and init the weights
    Args:
        dataset_name (str)
        model_name (str)
        loss_fuc_type (str)
        n_class (int): the number of class
        ckp_time_stamp (str): the checkpoints time stamp of model
        test_flag (bool): src_only, adda, and cyclegan+adda
    """
    # 定义模型结构
    if backbone_type == "resnet18" and loss_fuc_type == "softmax":
        cls_model = Resnet18_softmax(n_class)
        # ckp_time_stamp 不为空时加载预训练权重
        if ckp_time_stamp is not None:
            model_name = backbone_type + "_" + loss_fuc_type
            cls_model, arcface_loss_func = load_resnet18_softmax_ckp(
                cls_model, arcface_loss_func,
                dataset_name, model_name, ckp_time_stamp
            )
        
        return cls_model.cuda()
    elif backbone_type == "resnet18" and loss_fuc_type == "arcface":
        cls_model = Resnet18()
        arcface_loss_func = ArcFaceLoss(feat_dim=cls_model.feats_dim, n_class=n_class)

    elif backbone_type == "resnet18" and loss_fuc_type == "ElasticFacePlus-Arc":
        cls_model = Resnet18()
        arcface_loss_func = ElasticArcFace(
            in_features=cls_model.feats_dim, out_features=n_class, plus=True)
        
    elif backbone_type == "resnet18" and loss_fuc_type == "UniFaceArcFace":
        cls_model = Resnet18()
        arcface_loss_func = UniFaceArcFace(
            in_features=cls_model.feats_dim, out_features=n_class, l=0.6, r=0.6)
        
    else:
        raise RuntimeError(f"model_name:{backbone_type} is invalid")

    # ckp_time_stamp 不为空时加载预训练权重
    if ckp_time_stamp is not None:
        cls_model, arcface_loss_func = load_resnet18_arcface_ckp(
            cls_model, arcface_loss_func,
            dataset_name, backbone_type, loss_fuc_type, ckp_time_stamp
        )
    return cls_model.cuda(), arcface_loss_func.cuda()