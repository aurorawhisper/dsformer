from .dinov2 import DinoV2
from .resnet import ResNet

CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet50": [1024, 2048],
    "dinov2_vitb14": [768, 768]
}

def get_backbone(backbone_name='ResNet50', train_dsf=True):
    if backbone_name.startswith('ResNet'):
        return ResNet(backbone_name=backbone_name), CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    elif backbone_name.startswith('dinov2'):
        return DinoV2(backbone_name=backbone_name), CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    else:
        raise ValueError('Unsupported backbone')
