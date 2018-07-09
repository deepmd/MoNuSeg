from .res_unet_parts import *
from torchvision import models


# Transfer Learning ResNet as Encoder part of UNet
class Res_UNet(nn.Module):
    def __init__(self, layers=34, out_channels=1):
        super().__init__()
        # define pre-train model parameters
        if layers == 101:
            builder = models.resnet101
            l = [64, 256, 512, 1024, 2048]
        else:
            builder = models.resnet34
            l = [64, 64, 128, 256, 512]
        # load weight of pre-trained resnet
        self.resnet = builder(pretrained=True)
        # up conv
        self.u5 = ConvUpBlock(l[4], l[3])
        self.u6 = ConvUpBlock(l[3], l[2])
        self.u7 = ConvUpBlock(l[2], l[1])
        self.u8 = ConvUpBlock(l[1], l[0])
        # final conv tunnel
        self.ce = nn.ConvTranspose2d(l[0], out_channels, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        return x