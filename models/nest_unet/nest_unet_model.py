from .nest_unet_parts import *
from torchvision import models


class Nest_VGG_UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, in_channels=3, aux_out_channels=0, pretrained=False):
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
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.aux_out = nn.Conv2d(num_filters, 1, kernel_size=1) if aux_out_channels > 0 else None

        self.up1_1 = up(128, 64, 64, 2)
        self.up2_1 = up(256, 128, 128, 2)
        self.up3_1 = up(512, 256, 256, 2)
        self.up4_1 = up(512, 512, 512, 2)

        self.up3_2 = up(512, 256, 256, 2)
        self.up2_2 = up(256, 128, 128, 2)
        self.up1_2 = up(128, 64, 64, 2)

        self.up2_3 = up(256, 128, 128, 2)
        self.up1_3 = up(128, 64, 64, 2)

        self.up1_4 = up(128, 64, 64, 2)

    def forward(self, x):
        if x.shape[1] != 3:
            x = self.conv0(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        skip1_1 = self.up1_1(conv2, conv1)
        skip2_1 = self.up2_1(conv3, conv2)
        skip3_1 = self.up3_1(conv4, conv3)
        skip4_1 = self.up4_1(conv5, conv4)

        skip3_2 = self.up3_2(skip4_1, skip3_1)
        skip2_2 = self.up2_2(skip3_1, skip2_1)
        skip1_2 = self.up1_2(skip2_1, skip1_1)

        skip2_3 = self.up2_3(skip3_2, skip2_2)
        skip1_3 = self.up1_3(skip2_2, skip1_2)

        skip1_4 = self.up1_4(skip2_3, skip1_3)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, skip4_1], 1))
        dec3 = self.dec3(torch.cat([dec4, skip3_2], 1))
        dec2 = self.dec2(torch.cat([dec3, skip2_3], 1))
        dec1 = self.dec1(torch.cat([dec2, skip1_4], 1))

        # if self.num_classes > 1:
        #     x_out = F.log_softmax(self.final(dec1), dim=1)
        # else:
        #     x_out = self.final(dec1)

        x_out = self.final(dec1)
        x_aux_out = self.aux_out(dec1) if self.aux_out is not None else None

        return (x_out,x_aux_out) if x_aux_out is not None else x_out
