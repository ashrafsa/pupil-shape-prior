import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.max_pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = DoubleConv(in_channels, 64)
        self.down_conv_2 = DoubleConv(64, 128)
        self.down_conv_3 = DoubleConv(128, 256)
        self.down_conv_4 = DoubleConv(256, 512)
        self.down_conv_5 = DoubleConv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024,
                                             out_channels=512,
                                             kernel_size=2, stride=2)
        self.up_conv_1 = DoubleConv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512,
                                             out_channels=256,
                                             kernel_size=2, stride=2)
        self.up_conv_2 = DoubleConv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=2, stride=2)
        self.up_conv_3 = DoubleConv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=2, stride=2)
        self.up_conv_4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(in_channels=64,
                             out_channels=out_channels,
                             kernel_size=1)

    def __copy_and_crop(self, down_layer, up_layer):
        b, ch, h, w = up_layer.shape
        crop = CenterCrop((h, w))(down_layer)
        return crop

    def __crop_image(self, source: torch.TensorType, target: torch.TensorType):
        target_size = target.size()[2]
        source_size = source.size()[2]
        delta = source_size - target_size
        delta //= 2
        return source[:, :, delta:source_size - delta, delta: source_size - delta]

    def forward(self, x):
        # encoder
        x1 = self.down_conv_1(x)
        x2 = self.max_pool2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool2x2(x7)
        x9 = self.down_conv_5(x8)
        # x10 = self.max_pool2x2(x9)

        # decoder
        # step 1
        x = self.up_trans_1(x9)
        x = TF.resize(x, size=x7.shape[2:], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        x = self.up_conv_1(torch.cat([x, x7], dim=1))

        # step 2
        x = self.up_trans_2(x)
        x = TF.resize(x, size=x5.shape[2:], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        x = self.up_conv_2(torch.cat([x, x5], dim=1))

        # step 3
        x = self.up_trans_3(x)
        x = TF.resize(x, size=x3.shape[2:], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        x = self.up_conv_3(torch.cat([x, x3], dim=1))

        # step 4
        x = self.up_trans_4(x)
        x = TF.resize(x, size=x1.shape[2:], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        x = self.up_conv_4(torch.cat([x, x1], dim=1))

        return self.out(x)


def test():
    x = torch.randn((1, 1, 572, 572))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)

    print(f'Output {preds.shape} vs {x.shape}')

    assert preds.shape == x.shape


if __name__ == "__main__":
    test()