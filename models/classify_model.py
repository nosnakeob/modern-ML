from functools import partial

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchvision import models
from torchvision.models.swin_transformer import SwinTransformerBlockV2, PatchMergingV2


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5),  # B * 3 * 28 * 28 -> B * 16 * 24 * 24
            nn.MaxPool2d(2, 2),  # -> B * 16 * 12 * 12
            nn.Conv2d(16, 24, 3),  # -> B * 24 * 10 * 10
        )

        self.fc = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),  # B * 24 * 10 * 10 -> B * (24 * 10 * 10)
            nn.Linear(24 * 10 * 10, 480),  # B * 2400 -> B * 480
            nn.ReLU(),
            nn.Linear(480, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)

        return self.fc(x)


class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()

        self.trans = models.swin_v2_t(num_classes=10)

        # self.embed_dim = 96
        #
        # norm_layer = partial(nn.LayerNorm, eps=1e-5)
        #
        # self.patch_embed = nn.Sequential(
        #     nn.Conv2d(3, self.embed_dim, kernel_size=(4, 4), stride=(4, 4)),  # 3 * 28 * 28 -> 96 * 7 * 7
        #     Rearrange('b c h w -> b h w c'),
        #     norm_layer(self.embed_dim)  # 归一化
        # )
        #
        # self.basic_block = nn.Sequential(  # 7 * 7 * 96 -> 7 * 7 * 96
        #     SwinTransformerBlockV2(
        #         dim=self.embed_dim,
        #         num_heads=3,
        #         shift_size=[0, 0],
        #         window_size=[8, 8],
        #     ),
        #     SwinTransformerBlockV2(
        #         dim=self.embed_dim,
        #         num_heads=3,
        #         shift_size=[4, 4],
        #         window_size=[8, 8],
        #     ),
        # )
        #
        # self.patch_merge = PatchMergingV2(  # 7 * 7 * 96 -> 4 * 4 * 192
        #     dim=self.embed_dim,
        #     norm_layer=norm_layer
        # )
        #
        # self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.flatten = nn.Flatten(1)
        #
        # self.head = nn.Linear(192, 10)

    def forward(self, x):
        # x = self.patch_embed(x)  # B * 7 * 7 * 96
        # x = self.basic_block(x)  # B * 7 * 7 * 96
        # x = self.patch_merge(x)  # B * 4 * 4 * 192
        #
        # x = self.permute(x)
        # x = self.avgpool(x)
        # x = self.flatten(x)
        # x = self.head(x)

        x = self.trans(x)
        return x
