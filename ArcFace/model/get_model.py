
import torch
import torch.nn as nn
import os

from ArcFace.model.arcface import ArcFaceLoss
from ElasticFace.elasticface import ElasticArcFace
from UniFace.uniface import UniFaceArcFace
from global_utils import get_rootdir_path
from ArcFace.model.resnet import Resnet18, Resnet18_softmax


def load_model_ckp(cls_model:nn.Module, arcface_loss_func, 
                              dataset_name, backbone_type, loss_fuc_type, ckp_time_stamp):
    if loss_fuc_type == "softmax":
        parent_dir = "Softmax"
    elif loss_fuc_type == "arcface":
        parent_dir = "ArcFace"
    elif loss_fuc_type == "elasticfaceplusarc":
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
    print(f"ckp_path: {ckp_path} has been loaded!")

    # load archive
    archive = torch.load(ckp_path)
    
    cls_model.load_state_dict(archive["cls_model"])

    if arcface_loss_func is not None:
        arcface_loss_func.load_state_dict(
            archive[
                loss_fuc_type
            ]
        )
    return cls_model, arcface_loss_func



def defineModel(
        dataset_name, backbone_type, loss_fuc_type,
        n_class, 
        ckp_time_stamp=None
    )->nn.Module:
    """
    define the model structure and init the weights
    Args:
        dataset_name (str)
        model_name (str)
        loss_fuc_type (str)
        n_class (int): the number of class
        ckp_time_stamp (str): the checkpoints time stamp of model
    """
    if backbone_type == "resnet18" and loss_fuc_type == "softmax":
        cls_model = Resnet18_softmax(n_class)
        arcface_loss_func = None

    elif backbone_type == "resnet18" and loss_fuc_type == "arcface":
        cls_model = Resnet18()
        arcface_loss_func = ArcFaceLoss(feat_dim=cls_model.feats_dim, n_class=n_class)

    elif backbone_type == "resnet18" and loss_fuc_type == "elasticfaceplusarc":
        cls_model = Resnet18()
        arcface_loss_func = ElasticArcFace(
            in_features=cls_model.feats_dim, out_features=n_class, plus=True)
        
    elif backbone_type == "resnet18" and loss_fuc_type == "unifacearc":
        cls_model = Resnet18()
        arcface_loss_func = UniFaceArcFace(
            in_features=cls_model.feats_dim, out_features=n_class, l=0.6, r=0.6)
        
    else:
        raise RuntimeError(f"model_name:{backbone_type} is invalid")

    # load the pre-trained model weights 
    if ckp_time_stamp is not None:
        cls_model, arcface_loss_func = load_model_ckp(
            cls_model, arcface_loss_func,
            dataset_name, backbone_type, loss_fuc_type, ckp_time_stamp
        )
    if cls_model is not None:
        cls_model.cuda()
    if arcface_loss_func is not None:
        arcface_loss_func.cuda()
    return cls_model, arcface_loss_func