from .vgg_unet_parts import *
from torchvision import models


class VGG_Holistic_UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, in_channels=3, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, 3, 1),
                                   nn.BatchNorm2d(3),
                                   self.relu)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.outc5 = OutBlock(num_filters * 8, num_classes, 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.outc4 = OutBlock(num_filters * 8, num_classes, 4)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.outc3 = OutBlock(num_filters * 2, num_classes, 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.outc2 = OutBlock(num_filters, num_classes, 1)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.outc1 = OutBlock(num_filters, num_classes, 1)

        self.out_fuse = FuseBlock(5, num_classes)

    def forward(self, x):
        if x.shape[1] != 3:
            x = self.conv0(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        out5 = self.outc5(dec5)
        out4 = self.outc4(dec4)
        out3 = self.outc3(dec3)
        out2 = self.outc2(dec2)
        out1 = self.outc1(dec1)

        out_fuse = self.out_fuse(torch.cat([out1, out2, out3, out4, out5], 1))

        return out5, out4, out3, out2, out1, out_fuse


class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, is_deconv=True):
        super(OutBlock, self).__init__()
        self.in_channels = in_channels

        if scale_factor == 1:
            self.block = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif is_deconv:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor*2, stride=scale_factor,
                                   padding=scale_factor//2, output_padding=scale_factor%2),
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        return self.block(x)


class FuseBlock(nn.Module):
    def __init__(self, num_inputs, num_channels):
        super(FuseBlock, self).__init__()
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.orders = [i for j in range(num_channels) for i in range(j, num_channels*num_inputs, num_channels)]
        self.weight = nn.Parameter(torch.Tensor(num_channels*num_inputs, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[0])
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = x[:, self.orders].permute(0, 2, 3, 1)
        outputs = []
        for c in range(0, self.num_inputs*self.num_channels, self.num_inputs):
            outputs.append(x[..., c:c+self.num_inputs].matmul(self.weight[c:c+self.num_inputs]))

        return torch.cat(outputs, -1).permute(0, 3, 1, 2)

# separated weights for each pixel
# class FuseBlock(nn.Module):
#     def __init__(self, num_inputs, num_channels):
#         super(FuseBlock, self).__init__()
#         self.num_inputs = num_inputs
#         self.num_channels = num_channels
#         self.orders = [i for j in range(num_channels) for i in range(j, num_channels*num_inputs, num_channels)]
#         self.weight = nn.Parameter(torch.Tensor(num_channels*num_inputs, 128, 128))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.shape[0])
#         self.weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, x):
#         x = x[:, self.orders]
#         outputs = []
#         for c in range(0, self.num_inputs*self.num_channels, self.num_inputs):
#             outputs.append(torch.sum(x[:, c:c+self.num_inputs].mul(self.weight[c:c+self.num_inputs]), dim=1, keepdim=True))
#
#         return torch.cat(outputs, 1)
