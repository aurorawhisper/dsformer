import torch.nn as nn
import torchvision


class ResNet(nn.Module):
    def __init__(self, backbone_name='ResNet50'):
        super().__init__()
        assert backbone_name == 'ResNet50'
        self.backbone = getattr(torchvision.models, backbone_name.lower())()
        self.backbone.avgpool = None
        self.backbone.fc = None

    def forward(self, x):
        features = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features.append(x)
        x = self.backbone.layer4(x)
        features.append(x)
        return features







