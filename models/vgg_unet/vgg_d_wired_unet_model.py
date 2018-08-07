from .vgg_unet_parts import *
from ..unet.unet_parts import bottleneck
from .vgg_unet_model import VGG_UNet16


class VGG_DWired_UNet16(nn.Module):
    def __init__(self, config):
        super(VGG_DWired_UNet16, self).__init__()
        config1 = config['unet1']
        config2 = config['unet2']
        self.concat = config.get('concat', None)
        self.masking = config['masking']
        self.mask_dim = config['mask_dim']
        self.batch_norm = config.get('batch_norm', False)
        config2['in_channels'] = config1['out_channels'] if self.concat != 'input-discard_out1' else 0
        if self.concat is not None and 'input' in self.concat:
            config2['in_channels'] += config1['in_channels']
        self.unet1 = VGG_UNet16(in_channels=config1['in_channels'], num_classes=config1['out_channels'],
                                pretrained=config1['pretrained'])
        self.unet2 = VGG_UNet16(in_channels=config2['in_channels'], num_classes=config2['out_channels'],
                                pretrained=config2['pretrained'])
        self.bottleneck1 = bottleneck(64*2, 64)
        self.bottleneck2 = bottleneck(128*2, 128)
        self.bottleneck3 = bottleneck(256*2, 256)
        self.bottleneck4 = bottleneck(512*2, 512)
        self.bottleneck5 = bottleneck(512*2, 512)
        self.bn = nn.BatchNorm2d(config['unet2']['in_channels'])

    def forward(self, x):
        inp = x

        # VGG_UNet 1
        if x.shape[1] != 3:
            x = self.unet1.conv0(x)
        conv1_1 = self.unet1.conv1(x)
        conv2_1 = self.unet1.conv2(self.unet1.pool(conv1_1))
        conv3_1 = self.unet1.conv3(self.unet1.pool(conv2_1))
        conv4_1 = self.unet1.conv4(self.unet1.pool(conv3_1))
        conv5_1 = self.unet1.conv5(self.unet1.pool(conv4_1))

        center_1 = self.unet1.center(self.unet1.pool(conv5_1))

        dec5_1 = self.unet1.dec5(torch.cat([center_1, conv5_1], 1))

        dec4_1 = self.unet1.dec4(torch.cat([dec5_1, conv4_1], 1))
        dec3_1 = self.unet1.dec3(torch.cat([dec4_1, conv3_1], 1))
        dec2_1 = self.unet1.dec2(torch.cat([dec3_1, conv2_1], 1))
        dec1_1 = self.unet1.dec1(torch.cat([dec2_1, conv1_1], 1))
        output1 = self.unet1.final(dec1_1)

        # Processing output of unet1 to input of unet2
        if self.masking is not None:
            out1 = F.softmax(output1, dim=1)
            mask = (out1.argmax(dim=1, keepdim=True) == self.mask_dim).float() if self.masking == 'hard' else \
                   torch.unsqueeze(out1[:, self.mask_dim], dim=1)
            mask = mask.repeat((1, inp.shape[1], 1, 1))
            inp = inp * mask

        if self.concat == 'input-discard_out1':
            x = inp
        elif self.concat == 'input_softmax_out1':
            out1 = F.softmax(output1, dim=1)
            x = torch.cat([inp, out1], dim=1)
        elif self.concat == 'input':
            x = torch.cat([inp, output1], dim=1)
        elif self.concat == 'softmax_out1':
            x = F.softmax(output1, dim=1)
        else:
            x = output1

        if self.batch_norm:
            x = self.bn(x)

        # VGG_UNet 2
        if x.shape[1] != 3:
            x = self.unet2.conv0(x)
        conv1_2 = self.unet2.conv1(x)
        conv1_2 = self.bottleneck1(torch.cat([conv1_1, conv1_2], 1))
        conv2_2 = self.unet2.conv2(self.unet2.pool(conv1_2))
        conv2_2 = self.bottleneck2(torch.cat([conv2_1, conv2_2], 1))
        conv3_2 = self.unet2.conv3(self.unet2.pool(conv2_2))
        conv3_2 = self.bottleneck3(torch.cat([conv3_1, conv3_2], 1))
        conv4_2 = self.unet2.conv4(self.unet2.pool(conv3_2))
        conv4_2 = self.bottleneck4(torch.cat([conv4_1, conv4_2], 1))
        conv5_2 = self.unet2.conv5(self.unet2.pool(conv4_2))
        conv5_2 = self.bottleneck5(torch.cat([conv5_1, conv5_2], 1))

        center_2 = self.unet2.center(self.unet2.pool(conv5_2))

        dec5_2 = self.unet2.dec5(torch.cat([center_2, conv5_2], 1))

        dec4_2 = self.unet2.dec4(torch.cat([dec5_2, conv4_2], 1))
        dec3_2 = self.unet2.dec3(torch.cat([dec4_2, conv3_2], 1))
        dec2_2 = self.unet2.dec2(torch.cat([dec3_2, conv2_2], 1))
        dec1_2 = self.unet2.dec1(torch.cat([dec2_2, conv1_2], 1))
        output2 = self.unet2.final(dec1_2)

        return output1, output2
