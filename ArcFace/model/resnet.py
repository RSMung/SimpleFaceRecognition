import torchvision.models as vmodels
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary, ColumnSettings
import torch

    

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        # self.in_conv = nn.Conv2d(1, 3, 1)
        self.backbone = vmodels.resnet18(pretrained=True)
        # self.backbone = vmodels.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(
            in_features=self.backbone.fc.in_features,
            out_features=self.backbone.fc.in_features
        )
        self.feats_dim = self.backbone.fc.in_features

    def forward(self, x):
        # print(x.shape)
        # x = self.in_conv(x)
        x = self.backbone(x)
        return x
    

class Resnet18_softmax(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # self.in_conv = nn.Conv2d(1, 3, 1)
        self.backbone = vmodels.resnet18(pretrained=True)
        # self.backbone = vmodels.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(
            in_features=self.backbone.fc.in_features,
            out_features=n_class
        )
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # print(x.shape)
        # x = self.in_conv(x)
        x = self.backbone(x)
        x = self.softmax(x)
        return x
    

class Resnet34(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # self.in_conv = nn.Conv2d(1, 3, 1)
        self.backbone = vmodels.resnet34(pretrained=False)
        self.backbone.fc = nn.Linear(
            in_features=self.backbone.fc.in_features,
            out_features=n_class
        )
        # self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # print(x.shape)
        # x = self.in_conv(x)
        x = self.backbone(x)
        # x = self.softmax(x)
        return x
    

class Resnet50(nn.Module):
    def __init__(self, n_class) -> None:
        super().__init__()
        # self.in_conv = nn.Conv2d(1, 3, 1)
        # self.backbone = vmodels.resnet50(pretrained=True)
        self.backbone = vmodels.resnet50(pretrained=False)
        self.backbone.fc = nn.Linear(
            in_features=self.backbone.fc.in_features,
            out_features=n_class
        )
        # self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # print(x.shape)
        # x = self.in_conv(x)
        x = self.backbone(x)
        # x = self.softmax(x)
        return x
    
# a = vmodels.resnet18(pretrained=False)
# print(a)

# r = Resnet18(n_class=500)
# r = Resnet34(n_class=500)
# r = vmodels.resnet34()
# print(r)
# a = torch.randn((2, 3, 224, 224))
# # a = torch.randn((2, 1, 64, 64))
# summary(
#     r,
#     input_data=a,
#     # col_names=[
#     #     ColumnSettings.INPUT_SIZE,
#     #     ColumnSettings.OUTPUT_SIZE,
#     #     ColumnSettings.NUM_PARAMS
#     # ]
# )
