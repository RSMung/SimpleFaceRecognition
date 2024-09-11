import torchvision.models as vmodels
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary, ColumnSettings
import torch

    

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vmodels.resnet18(pretrained=True)
        # self.backbone = vmodels.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(
            in_features=self.backbone.fc.in_features,
            out_features=self.backbone.fc.in_features
        )
        self.feats_dim = self.backbone.fc.in_features

    def forward(self, x):
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

    def get_feats(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.softmax(x)
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
