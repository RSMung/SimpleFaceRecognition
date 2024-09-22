import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from EasyLossUtil.global_utils import ParamsParent, checkDir, formatSeconds
from EasyLossUtil.easyLossUtil import EasyLossUtil

import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from tqdm import tqdm
import sys
import signal
from datetime import timedelta

from ArcFace.model.get_model import defineModel
from global_utils import identification_procedure, prepareEnv, verification_procedure
from load_data.get_dataloader import getDataloader



def build_src_ckp_path(ckp_time_stamp, dataset_name, model_name):
    assert ckp_time_stamp is not None
    assert dataset_name is not None
    assert model_name is not None
    # 当前文件的路径
    current_path = os.path.abspath(__file__)
    # 当前文件所在的目录
    root_dir = os.path.dirname(current_path)
    # the path for saving model checkpoints
    ckp_root_path = os.path.join(root_dir, "ckp", ckp_time_stamp)
    checkDir(ckp_root_path)
    cls_model_model_ckp_path = os.path.join(
        ckp_root_path,
        dataset_name + "_" + model_name + "_" + ckp_time_stamp
    )
    # the path for saving loss data
    loss_root_path = os.path.join(root_dir, "loss", "loss_" + ckp_time_stamp)
    return cls_model_model_ckp_path, loss_root_path



class TrainUniFaceParams(ParamsParent):
    # gpu_id = 0
    gpu_id = 1
    # gpu_id = 2
    # gpu_id = 3

    #region mnist
    # dataset_name = "mnist"
    # n_class = 10
    # img_size = 128
    # proportion = None   # mnist 数据集不需要比例参数, 默认 50000:10000:10000
    #endregion

    # dataset_name = "cifar100"
    # n_class = 100
    # img_size = 128
    # proportion = None   # cifar100 数据集不需要比例参数, 默认 40000:10000:10000

    dataset_name = "cifar10"
    n_class = 10
    img_size = 128
    proportion = None   # cifar10 数据集不需要比例参数, 默认 40000:10000:10000

    # batch_size = 48
    batch_size = 128
    lr = 5e-5

    total_epochs = 1000
    # total_epochs = 1
    early_stop_epochs = 20
    # early_stop_epochs = 50

    backbone_type = "resnet18"
    loss_fuc_type = "UniFaceArcFace"
    # loss_fuc_type = "softmax"
    model_name = backbone_type + "_" + loss_fuc_type

    use_tqdm = False
    # use_tqdm = True

    # 是否快速调试
    quick_debug = False
    # quick_debug = True

    # 是否在测试模型时 保存DET曲线数据到csv文件中
    save_csv = False
    # save_csv = True

    # ckp_time_stamp = "2024-09-21_14-00"   # 实验 7   cifar100   resnet18   unifacearc
    ckp_time_stamp = "2024-09-21_14-01"   # 实验 8   cifar10   resnet18   unifacearc
    
    model_ckp_path, loss_root_path = build_src_ckp_path(
        ckp_time_stamp,
        dataset_name,
        model_name
    )

    # nohup python -u main.py > ./UniFace/log/2024-09-13_17-00.txt 2>&1 &
    # 实验 7      1750562
    # nohup python -u main.py > ./UniFace/log/2024-09-13_17-01.txt 2>&1 &
    # 实验 8      1750738





def train_procedure(
        params:TrainUniFaceParams,
        cls_model:nn.Module, unifacearc_loss_func:nn.Module,
        train_dataloader,
        val_dataloader,
        # test_dataloader
    ):
    """
    the train procedure for training normal classifier model
    Args:
        params (TrainNormalClsParams): all the parameters
        cls_model (nn.Module): the model we want to train
        unifacearc_loss_func (nn.Module): unifacearc_loss_func
        train_dataloader: training data
        val_dataloader: validating data
    """
    # ------------------------------
    # -- init tool
    # ------------------------------
    loss_name = ['avg_train_batch_loss', 'val_loss', 'val_acc', 'val_eer']
    # 数据可视化处理工具
    lossUtil = EasyLossUtil(
        loss_name_list=loss_name,
        loss_root_dir=params.loss_root_path
    )

    # -----------------------------------
    # --  setup optimizer and lr_scheduler
    # -----------------------------------
    optimizer = optim.AdamW(
    # optimizer = optim.RMSprop(
        # vgg16_2 only update the params of fc module
        cls_model.parameters(),
        lr=params.lr,
        # weight_decay=0.05
        weight_decay=5e-4
    )
    # Decay LR by a factor of 0.1 every 7 epochs
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

    # ------------------------------
    # --  setup loss function
    # ------------------------------
    ce_loss_func = nn.CrossEntropyLoss()

    #region kill process
    # ----------------------------------------------------------------------
    # 定义一个信号处理函数, 用于处理kill命令的SIGTERM信号, 在退出前保存一次模型
    # ----------------------------------------------------------------------
    def handle_sigterm(signum, frame):
        """
        当使用 kill <进程id> 命令终止程序时会调用这个程序
        当使用 ctrl+c 命令终止程序时会调用这个程序
        Args:
            signum: 一个整数，代表接收到的信号的编号. signal.SIGTERM
            frame: 一个包含有关信号的堆栈信息的对象
        """
        if signum == signal.SIGTERM:
            print("Received kill signal. Performing saving procedure for ckps...")
        elif signum == signal.SIGINT:
            print("Received Ctrl+C signal. Performing saving procedure for ckps...")
        torch.save(
            {"cls_model":cls_model.state_dict(), "UniFaceArcFace":unifacearc_loss_func.state_dict()}, 
            params.model_ckp_path + "_kill"
        )
        print(f"Saving procedure completed: {params.model_ckp_path}_kill")
        sys.exit(0)

    # 注册SIGTERM信号处理函数
    # 将handle_sigterm函数与特定的信号（signal.SIGTERM）相关联
    signal.signal(signal.SIGTERM, handle_sigterm)
    # SIGINT是Ctrl+C传送的中断进程信号
    signal.signal(signal.SIGINT, handle_sigterm)
    #endregion

    # ------------------------------
    # --  train the model
    # ------------------------------
    min_val_loss = None
    no_change_epochs = 0
    avg_train_batch_loss = 0
    batch_num = 0
    for epoch in range(params.total_epochs):
        start_time = time.time()
        # setup network
        cls_model.train()
        unifacearc_loss_func.train() 
        # setup the progress bar
        if params.use_tqdm:
            iter_object = tqdm(train_dataloader, ncols=100)
        else:
            iter_object = train_dataloader
        for step, (images, labels) in enumerate(iter_object):
            if params.quick_debug:
                if step > 3:
                    break
            # print(images.shape)
            # print(labels.shape)
            # send images to gpu
            images = images.cuda()
            labels = labels.cuda()

            # ----------------------------------------------------------
            # - 首先从标签中提取出当前 batch 中已经出现的类别（positive）
            # - 然后从剩余类别中随机选取一定比例的类别
            # ----------------------------------------------------------
            # 提取出不重复的标签值, 并且按升序排列
            positive = torch.unique(labels, sorted=True)
            # 生成一个包含从 0 到 params.n_class-1 的整数随机排列
            perm = torch.randperm(params.n_class)
            # 本次batch中出现的标签的位置 置0
            perm[positive] = 0
            # 获取batch中没出现的标签中最小的k个标签
            # 为了平衡正负样本的数量
            # torch.topk 的返回值是一个二元组 (values, indices)
            # values 是 perm 中最小的 k 个元素
            # indices 是这些元素在 perm 中对应的索引位置
            indices = torch.topk(
                perm, 
                k=int(params.n_class*unifacearc_loss_func.r), 
                largest=False
            )[1]
            # 对 indices 中的元素进行升序排序，并返回排序后的张量 partial_index
            partial_index = indices.sort()[0]
            partial_index = partial_index.cuda()

            # zero gradients for optimizer
            optimizer.zero_grad()

            # infer
            feats = cls_model(images)

            # computing loss
            batch_loss = unifacearc_loss_func(feats, labels, partial_index)

            # record loss
            avg_train_batch_loss += batch_loss.item()
            batch_num += 1

            # update parameters
            batch_loss.backward()
            optimizer.step()
        # end for iter_object

        # one epoch end

        avg_train_batch_loss /= batch_num

        # print("-----validate model on training set-----")
        # # validate model on training set
        # train_loss, train_acc, train_eer = validate_procedure(
        #         cls_model, arcface_loss_func,
        #         train_dataloader, train_dataloader, 
        #         params
        # )

        # print("-----validate model on validation set-----")
        # validate model on validation set
        val_loss, val_acc, val_eer = validate_procedure(
                cls_model, unifacearc_loss_func,
                train_dataloader, val_dataloader, 
                params
        )

        # print("-----validate model on test set-----")
        # # validate model on test set
        # test_loss, test_acc, test_eer = validate_procedure(
        #         cls_model, arcface_loss_func,
        #         train_dataloader, test_dataloader, 
        #         params
        # )

        # adjust the learning rate
        step_lr_scheduler.step()

        # display train log
        print(
            f'[{epoch}/{params.total_epochs}]\n'
            f'avg_train_batch_loss: {avg_train_batch_loss:.4f}\n '
            # f'train_acc: {train_acc*100:.2f} %, '
            # f'train_eer: {train_eer*100:.2f} %\n'
            f'val_loss: {val_loss:.4f} , '
            f'val_acc: {val_acc*100:.2f} %, '
            f'val_eer: {val_eer*100:.2f} %\n'
            # f'test_loss: {test_loss:.4f} , '
            # f'test_acc: {test_acc*100:.2f} %, '
            # f'test_eer: {test_eer*100:.2f} %'
        )

        # print(f"avg_train_batch_loss:{avg_train_batch_loss.device}")
        # print(f"val_loss:{val_loss.device}")
        # print(f"val_acc:{val_acc.device}")
        # print(f"val_eer:{val_eer.device}")
        # save the log data
        lossUtil.append(
            loss_name=loss_name,
            loss_data=[
                avg_train_batch_loss, 
                # train_acc, train_eer,
                val_loss, val_acc, val_eer,
                # test_loss, test_acc, test_eer
            ]
        )
        lossUtil.autoSaveFileAndImage()

        # save model parameters
        no_change_epochs += 1
        if min_val_loss is None or val_loss < min_val_loss:
            torch.save(
                {
                    "cls_model":cls_model.state_dict(), 
                    "UniFaceArcFace":unifacearc_loss_func.state_dict()
                }, 
                params.model_ckp_path
            )
            min_val_loss = val_loss
            no_change_epochs = 0
            print('已经保存当前模型')
        
        # the time for this epoch
        end_time = time.time()
        print(f"epoch time cost:   {str(timedelta(seconds=int(end_time-start_time)))}")
        print()

        # early stop
        if no_change_epochs > params.early_stop_epochs:
            print('early stop train model')
            break

    # end all epoch
    return cls_model, unifacearc_loss_func


@torch.no_grad()
def validate_procedure(
    cls_model:nn.Module, 
    unifacearc_loss_func:nn.Module,
    train_dataloader:DataLoader,
    query_dataloader:DataLoader, 
    params:TrainUniFaceParams,
    save_csv=False
):
    """
    验证环节, 计算模型在指定数据集上的性能
    Args:
        cls_model (nn.Module): 特征提取器
        unifacearc_loss_func (nn.Module): unifacearc_loss_func 模块, 包含最后一个fc层, 可以用于计算acc
        train_dataloader (DataLoader): 训练集, 即注册集
        query_dataloader (DataLoader): 指定的数据集
        params (TrainUniFaceParams): 外部参数
    """
    # print("obtain the features and labels of training data")
    train_feats, train_labels = get_feats_labels(cls_model, train_dataloader, params)
    # print("obtain the features and labels of query data")
    query_feats, query_labels = get_feats_labels(cls_model, query_dataloader, params)

    # print("get the identification accuracy on train data")
    acc = identification_procedure(
        train_feats, train_labels, 
        query_feats, query_labels
    )
    # print("get the eer on train data")
    eer = verification_procedure(
        train_feats, train_labels, 
        query_feats, query_labels,
        save_csv=save_csv
    )
    # print("get the value of cross entropy loss")
    unifacearc_loss = get_unifacearc_loss(
        cls_model, unifacearc_loss_func, 
        query_dataloader, params
    )
    return unifacearc_loss, acc, eer


@torch.no_grad()
def get_feats_labels(cls_model:nn.Module, dataloader:DataLoader, params:TrainUniFaceParams):
    # setup the model on eval mode
    cls_model.eval()

    # setup the dataloader
    if params.use_tqdm:
        iter_object = tqdm(dataloader, ncols=100)
    else:
        iter_object = dataloader
    
    all_feats = []
    all_labels = []

    # loop
    for (images, labels) in iter_object:
        images = images.cuda()
        labels = labels.cuda()

        feats = cls_model(images)
        all_feats.append(feats)
        all_labels.append(labels)
    # end for dataloader

    # list to tensor
    all_feats = torch.concat([item for item in all_feats])
    all_labels = torch.concat([item for item in all_labels])
    return all_feats, all_labels


@torch.no_grad()
def get_unifacearc_loss(cls_model:nn.Module, unifacearc_loss_func:nn.Module, dataloader:DataLoader, params:TrainUniFaceParams):
    # setup the model on eval mode
    cls_model.eval()
    unifacearc_loss_func.eval()
    # last_fc = unifacearc_loss_func.weight

    # setup the dataloader
    if params.use_tqdm:
        iter_object = tqdm(dataloader, ncols=100)
    else:
        iter_object = dataloader

    # ce_entropy_loss = nn.CrossEntropyLoss()

    ce_loss = 0
    batch_num = 0

    # loop
    for (images, labels) in iter_object:
        images = images.cuda()
        labels = labels.cuda()

        # ----------------------------------------------------------
        # - 首先从标签中提取出当前 batch 中已经出现的类别（positive）
        # - 然后从剩余类别中随机选取一定比例的类别
        # ----------------------------------------------------------
        # 提取出不重复的标签值, 并且按升序排列
        positive = torch.unique(labels, sorted=True)
        # 生成一个包含从 0 到 params.n_class-1 的整数随机排列
        perm = torch.randperm(params.n_class)
        # 本次batch中出现的标签的位置 置0
        perm[positive] = 0
        # 获取batch中没出现的标签中最小的k个标签
        # 为了平衡正负样本的数量
        # torch.topk 的返回值是一个二元组 (values, indices)
        # values 是 perm 中最小的 k 个元素
        # indices 是这些元素在 perm 中对应的索引位置
        indices = torch.topk(
            perm, 
            k=int(params.n_class*unifacearc_loss_func.r), 
            largest=False
        )[1]
        # 对 indices 中的元素进行升序排序，并返回排序后的张量 partial_index
        partial_index = indices.sort()[0]
        partial_index = partial_index.cuda()

        feats = cls_model(images)
        batch_loss = unifacearc_loss_func(feats, labels, partial_index)

        ce_loss += batch_loss.item()
        batch_num += 1
    # end for dataloader

    ce_loss /= batch_num
    return ce_loss



def trainUniFaceMain():
    # ------------------------------
    # -- init the env
    # ------------------------------
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(TrainUniFaceParams.gpu_id)
    prepareEnv()
    # ------------------------------
    # -- init src train params
    # ------------------------------
    params = TrainUniFaceParams()
    print(params)
    # ------------------------------
    # -- load data
    # ------------------------------
    print("=== load data ===")
    train_dataloader = getDataloader(
        params.dataset_name, 
        phase="train", 
        img_size=params.img_size, 
        batch_size=params.batch_size,
        proportion=params.proportion
    )
    print(len(train_dataloader.dataset))
    val_dataloader = getDataloader(
        params.dataset_name, 
        "val", 
        img_size=params.img_size, 
        batch_size=params.batch_size,
        proportion=params.proportion
    )
    print(len(val_dataloader.dataset))
    test_dataloader = getDataloader(
        params.dataset_name, 
        "test", 
        img_size=params.img_size, 
        batch_size=params.batch_size,
        proportion=params.proportion
    )
    print(len(test_dataloader.dataset))

    # ------------------------------------------
    # -- define the structure of src model
    # ------------------------------------------
    print("=== init model ===")
    # params.dataset_name, params.model_name, params.n_class
    cls_model, unifacearc_loss_func = defineModel(
        params.dataset_name, params.backbone_type, params.loss_fuc_type, params.n_class)
    print(type(cls_model))
    print(type(unifacearc_loss_func))

    # ------------------------------
    # -- train model
    # ------------------------------
    print("=== train model ===")
    cls_model, unifacearc_loss_func = train_procedure(
        params,
        cls_model, unifacearc_loss_func,
        train_dataloader, val_dataloader, 
        # test_dataloader
    )

    # ------------------------------
    # -- test best model
    # ------------------------------
    print("-------------------------------------")
    print("- Test best model on test dataset")
    print("-------------------------------------")
    # load archive
    archive = torch.load(params.model_ckp_path)
    # 加载进模型
    cls_model.load_state_dict(archive["cls_model"])
    unifacearc_loss_func.load_state_dict(archive["UniFaceArcFace"])
    # setup mode
    cls_model.eval()
    unifacearc_loss_func.eval()
    # validate model on test set
    test_loss, test_acc, test_eer = validate_procedure(
        cls_model, unifacearc_loss_func,
        train_dataloader, test_dataloader, 
        params,
        save_csv=params.save_csv
    )
    print(
        f'test_loss:{test_loss:.4f}, '
        f'test_acc: {test_acc*100:.2f} %, '
        f'test_eer: {test_eer*100:.2f} %'
    )