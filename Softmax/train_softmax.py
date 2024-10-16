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



class TrainSoftmaxParams(ParamsParent):

    #region mnist
    # dataset_name = "mnist"
    # n_class = 10
    # img_size = 128
    # proportion = None   # default train:val:test = 50000:10000:10000
    #endregion

    #region cifar100
    dataset_name = "cifar100"
    n_class = 100
    img_size = 128
    proportion = None   # default train:val:test = 40000:10000:10000
    #endregion

    #region cifar10
    # dataset_name = "cifar10"
    # n_class = 10
    # img_size = 128
    # proportion = None   # default train:val:test = 40000:10000:10000
    #endregion

    # batch_size = 48
    batch_size = 128
    lr = 5e-5

    total_epochs = 1000
    # total_epochs = 1
    early_stop_epochs = 20

    backbone_type = "resnet18"
    # loss_fuc_type = "arcface"
    loss_fuc_type = "softmax"
    model_name = backbone_type + "_" + loss_fuc_type

    # if this flag is True:
    # the progress bar will be shown
    use_tqdm = False
    # use_tqdm = True

    # if this flag is true:
    # the for loop will be broken after few steps
    quick_debug = False
    # quick_debug = True

    # if this flag is True:
    # the DET curve data will be saved when we calculate the eer
    save_csv = False
    # save_csv = True

    ckp_time_stamp = "2024-10-15_16-00"   # exp1   cifar100   resnet18   softmax
    
    model_ckp_path, loss_root_path = build_src_ckp_path(
        ckp_time_stamp,
        dataset_name,
        model_name
    )

    # nohup python -u main.py > ./Softmax/log/2024-10-15_16-00.txt 2>&1 &
    # exp1      703126





def train_procedure(
        params:TrainSoftmaxParams,
        cls_model:nn.Module, 
        train_dataloader,
        val_dataloader,
    ):
    """
    the train procedure for training normal classifier model
    Args:
        params (TrainSoftmaxParams): all the parameters
        cls_model (nn.Module): the model we want to train
        train_dataloader: training data
        val_dataloader: validating data
    """
    # ------------------------------
    # -- init tool
    # ------------------------------
    loss_name = ['avg_train_batch_loss', 'val_loss', 'val_acc', 'val_eer']
    # the util for visualization loss data
    lossUtil = EasyLossUtil(
        loss_name_list=loss_name,
        loss_root_dir=params.loss_root_path
    )

    # --------------------------------------
    # --  setup optimizer and lr_scheduler
    # --------------------------------------
    optimizer = optim.AdamW(
    # optimizer = optim.RMSprop(
        cls_model.parameters(),
        lr=params.lr,
        # weight_decay=0.05
        weight_decay=5e-4
    )
    # Decay LR by a factor of 0.99 every 10 epochs
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

    # ------------------------------
    # --  setup loss function
    # ------------------------------
    ce_loss_func = nn.CrossEntropyLoss()

    #region kill process
    # ---------------------------------------------------------------------------------------
    # Define a signal processing function to handle the SIGTERM signal of the kill command and 
    # save the model before exiting
    # ---------------------------------------------------------------------------------------
    def handle_sigterm(signum, frame):
        """
        When using the "kill <process id>" and "ctrl+c" to terminate a program, 
        this function will be called
        Args:
            signum: An integer representing the number of the received signal. signal.SIGTERM
            frame: An object containing stack information about signals
        """
        if signum == signal.SIGTERM:
            print("Received kill signal. Performing saving procedure for ckps...")
        elif signum == signal.SIGINT:
            print("Received Ctrl+C signal. Performing saving procedure for ckps...")
        torch.save(
            {"cls_model":cls_model.state_dict()}, 
            params.model_ckp_path + "_kill"
        )
        print(f"Saving procedure completed: {params.model_ckp_path}_kill")
        sys.exit(0)

    # Register SIGTERM signal processing function
    # Associate the handle_sterm function with a specific signal (signal.SIGTERM)
    signal.signal(signal.SIGTERM, handle_sigterm)
    # SIGINT is an interrupt process signal transmitted by Ctrl+C
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
        # setup the progress bar
        if params.use_tqdm:
            iter_object = tqdm(train_dataloader, ncols=100)
        else:
            iter_object = train_dataloader
        for step, (images, labels) in enumerate(iter_object):

            if params.quick_debug:
                if step > 3:
                    break
                
            # send images to gpu
            images = images.cuda()
            labels = labels.cuda()

            # zero gradients for optimizer
            optimizer.zero_grad()

            # infer
            logits = cls_model(images)

            # 计算损失值
            batch_loss = ce_loss_func(logits, labels)

            # 记录损失值
            avg_train_batch_loss += batch_loss.item()
            batch_num += 1

            # 更新参数
            batch_loss.backward()
            optimizer.step()
        # end for iter_object

        # one epoch end

        avg_train_batch_loss /= batch_num

        # print("-----validate model on validation set-----")
        # validate model on validation set
        val_loss, val_acc, val_eer = validate_procedure(
                cls_model,
                train_dataloader, val_dataloader, 
                params
        )

        # adjust the learning rate
        step_lr_scheduler.step()

        # display train log
        print(
            f'[{epoch}/{params.total_epochs}]\n'
            f'avg_train_batch_loss: {avg_train_batch_loss:.4f}\n '
            f'val_loss: {val_loss:.4f} , '
            f'val_acc: {val_acc*100:.2f} %, '
            f'val_eer: {val_eer*100:.2f} %'
        )

        # save the log data
        lossUtil.append(
            loss_name=loss_name,
            loss_data=[
                avg_train_batch_loss, 
                val_loss, val_acc, val_eer,
            ]
        )
        lossUtil.autoSaveFileAndImage()

        # save model parameters
        no_change_epochs += 1
        if min_val_loss is None or val_loss < min_val_loss:
            torch.save(
                {"cls_model":cls_model.state_dict()}, 
                params.model_ckp_path
            )
            min_val_loss = val_loss
            no_change_epochs = 0
            print('The weights of the model have been saved!')
        
        # the time for this epoch
        end_time = time.time()
        print(f"epoch time cost:   {str(timedelta(seconds=int(end_time-start_time)))}")
        print()

        # early stop
        if no_change_epochs > params.early_stop_epochs:
            print('early stop train model')
            break

    # end all epoch
    return cls_model


@torch.no_grad()
def validate_procedure(
    cls_model:nn.Module, 
    train_dataloader:DataLoader,
    query_dataloader:DataLoader, 
    params:TrainSoftmaxParams,
    save_csv=False
):
    """
    Verification phase, calculating the performance of the model on a specified dataset
    Args:
        cls_model (nn.Module): model
        train_dataloader (DataLoader): Training set, also known as registration set
        query_dataloader (DataLoader): specific dataset
        params (TrainSoftmaxParams): External parameters
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
    ce_loss = get_cross_entropy_loss(
        cls_model, 
        query_dataloader, params
    )
    return ce_loss, acc, eer


@torch.no_grad()
def get_feats_labels(cls_model:nn.Module, dataloader:DataLoader, params:TrainSoftmaxParams):
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

        feats = cls_model.get_feats(images)
        all_feats.append(feats)
        all_labels.append(labels)
    # end for dataloader

    # list to tensor
    all_feats = torch.concat([item for item in all_feats])
    all_labels = torch.concat([item for item in all_labels])
    return all_feats, all_labels


@torch.no_grad()
def get_cross_entropy_loss(cls_model:nn.Module, dataloader:DataLoader, params:TrainSoftmaxParams):
    # setup the model on eval mode
    cls_model.eval()

    # setup the dataloader
    if params.use_tqdm:
        iter_object = tqdm(dataloader, ncols=100)
    else:
        iter_object = dataloader

    ce_entropy_loss = nn.CrossEntropyLoss()

    ce_loss = 0
    batch_num = 0

    # loop
    for (images, labels) in iter_object:
        images = images.cuda()
        labels = labels.cuda()

        logits = cls_model(images)
        batch_loss = ce_entropy_loss(logits, labels)

        ce_loss += batch_loss.item()
        batch_num += 1
    # end for dataloader

    ce_loss /= batch_num
    return ce_loss



def trainSoftmaxMain():
    # ------------------------------
    # -- init the env
    # ------------------------------
    prepareEnv()
    # ------------------------------
    # -- init src train params
    # ------------------------------
    params = TrainSoftmaxParams()
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
    cls_model, _ = defineModel(params.dataset_name, params.backbone_type, params.loss_fuc_type, params.n_class)
    print(type(cls_model))

    # ------------------------------
    # -- train model
    # ------------------------------
    print("=== train model ===")
    cls_model = train_procedure(
        params,
        cls_model,
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
    cls_model.load_state_dict(
        torch.load(params.model_ckp_path)["cls_model"]
    )
    # setup mode
    cls_model.eval()
    # validate model on test set
    test_loss, test_acc, test_eer = validate_procedure(
        cls_model,
        train_dataloader, test_dataloader, 
        params,
        save_csv=params.save_csv
    )
    print(
        f'test_loss:{test_loss:.4f}, '
        f'test_acc: {test_acc*100:.2f} %, '
        f'test_eer: {test_eer*100:.2f} %'
    )