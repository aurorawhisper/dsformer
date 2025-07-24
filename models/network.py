
import torch.nn as nn
import torch
from models.backbone import get_backbone
from models.utils import L2Norm, GeM, Flatten
from models.transformer import TransformerSelfEncoderLayer, TransformerCrossEncoderLayer

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class DualScaleFormer(nn.Module):
    def __init__(self, backbone_feature_dims=[1024, 2048],
                 num_patches=[400, 100],
                 embedding_dim=1024,
                 num_heads=16,
                 mlp_ratio=4.,
                 bias=True,
                 num_layers=3,
                 drop_ratio=0.):
        super().__init__()
        self.norm_layer_0 = nn.LayerNorm(embedding_dim)
        self.norm_layer_1 = nn.LayerNorm(embedding_dim)
        self.pos_embed0 = nn.Parameter(torch.randn(1, num_patches[0], embedding_dim) * .02)
        self.pos_embed1 = nn.Parameter(torch.randn(1, num_patches[1], embedding_dim) * .02)
        self.num_layers = num_layers
        if backbone_feature_dims[0] == embedding_dim:
            self.input_proj_0 = nn.Identity()
        else:
            self.input_proj_0 = nn.Linear(in_features=backbone_feature_dims[0],
                                          out_features=embedding_dim,
                                          bias=bias)

        if backbone_feature_dims[1] == embedding_dim:
            self.input_proj_1 = nn.Identity()
        else:
            self.input_proj_1 = nn.Linear(in_features=backbone_feature_dims[1],
                                          out_features=embedding_dim,
                                          bias=bias)
        self.TransformerSelfEncoderLayer0 = nn.ModuleList(
            [TransformerSelfEncoderLayer(embedding_dim=embedding_dim,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=bias,
                                         act_layer=nn.GELU,
                                         norm_layer=nn.LayerNorm,
                                         use_irpe=True,
                                         drop_ratio=drop_ratio) for _ in range(num_layers)])
        self.TransformerSelfEncoderLayer1 = nn.ModuleList(
            [TransformerSelfEncoderLayer(embedding_dim=embedding_dim,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=bias,
                                         act_layer=nn.GELU,
                                         norm_layer=nn.LayerNorm,
                                         use_irpe=True,
                                         drop_ratio=drop_ratio) for _ in range(num_layers)])

        self.TransformerCrossEncoderLayer = nn.ModuleList(
            [TransformerCrossEncoderLayer(embedding_dim=embedding_dim,
                                          num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=bias,
                                          act_layer=nn.GELU,
                                          norm_layer=nn.LayerNorm,
                                          use_irpe=False,
                                          drop_ratio=drop_ratio) for _ in range(num_layers)])

    def forward(self, x0, x1):
        x0 = x0.flatten(2).permute(0, 2, 1).contiguous()
        x1 = x1.flatten(2).permute(0, 2, 1).contiguous()
        x0 = self.input_proj_0(x0)
        x1 = self.input_proj_1(x1)
        x0 = x0 + self.pos_embed0
        x1 = x1 + self.pos_embed1
        for i in range(self.num_layers):
            x0 = self.TransformerSelfEncoderLayer0[i](x0)
            x1 = self.TransformerSelfEncoderLayer1[i](x1)
            x0, x1 = self.TransformerCrossEncoderLayer[i](x0, x1)
        x0 = self.norm_layer_0(x0)
        x1 = self.norm_layer_1(x1)
        return x0, x1


class GeoLocalizationNet(nn.Module):

    def __init__(self, backbone='ResNet50',
                 num_patches=[400, 100],
                 mlp_ratio=4.,
                 bias=True,
                 num_layers=3,
                 drop_ratio=0.,
                 fc_output_dim=512):
        super().__init__()
        self.backbone, input_dim = get_backbone(backbone)
        embedding_dim = min(input_dim)
        self.dual_scale_former = DualScaleFormer(backbone_feature_dims=input_dim,
                                                 num_patches=num_patches,
                                                 embedding_dim=embedding_dim,
                                                 num_heads=embedding_dim//64,
                                                 mlp_ratio=mlp_ratio,
                                                 bias=bias,
                                                 num_layers=num_layers,
                                                 drop_ratio=drop_ratio)

        self.aggregation0 = nn.Sequential(L2Norm(), GeM(), Flatten())
        self.aggregation1 = nn.Sequential(L2Norm(), GeM(), Flatten())
        self.fc = nn.Linear(in_features=embedding_dim * 2,
                            out_features=fc_output_dim,
                            bias=bias)
        self.l2_norm = L2Norm()
    def forward(self, x):
        x = self.backbone(x)
        B, _, H0, W0 = x[0].shape
        _, _, H1, W1 = x[1].shape
        x0, x1 = self.dual_scale_former(x[0], x[1])
        xg0 = x0.view(B, H0, W0, -1)
        xg0 = xg0.permute(0, 3, 1, 2)
        xg1 = x1.view(B, H1, W1, -1)
        xg1 = xg1.permute(0, 3, 1, 2)
        x = torch.cat([self.aggregation0(xg0), self.aggregation1(xg1)], dim=-1)
        x = self.fc(x)
        x = self.l2_norm(x)
        return x












