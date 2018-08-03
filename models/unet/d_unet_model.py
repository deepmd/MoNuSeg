from .unet_parts import *
from ..unet import UNet
from ..vgg_unet import VGG_UNet11, VGG_UNet16


class DUNet(nn.Module):
    def __init__(self, config):
        super(DUNet, self).__init__()
        config1 = config['unet1']
        config2 = config['unet2']
        self.concat = config.get('concat', None)
        self.masking = config['masking']
        self.mask_dim = config['mask_dim']
        self.batch_norm = config.get('batch_norm', False)
        config2['in_channels'] = config1['out_channels'] if self.concat != 'input-discard_out1' else 0
        if self.concat is not None and 'input' in self.concat:
            config2['in_channels'] += config1['in_channels']
        self.unet1 = UNet(config1) if 'vgg' not in config1 else \
            (VGG_UNet11(num_classes=config1['out_channels'], pretrained=config1['pretrained']) if config1['vgg'] == 11 else \
             VGG_UNet16(num_classes=config1['out_channels'], pretrained=config1['pretrained']))
        self.unet2 = UNet(config2) if 'vgg' not in config2 else \
            (VGG_UNet11(num_classes=config2['out_channels'], pretrained=config2['pretrained']) if config2['vgg'] == 11 else \
             VGG_UNet16(num_classes=config2['out_channels'], pretrained=config2['pretrained']))
        self.bn = nn.BatchNorm2d(config['unet2']['in_channels'])

    def forward(self, x):
        output1 = self.unet1(x)

        if self.masking is not None:
            out1 = F.softmax(output1, dim=1)
            mask = (out1.argmax(dim=1, keepdim=True) == self.mask_dim).float() if self.masking == 'hard' else \
                   torch.unsqueeze(out1[:, self.mask_dim], dim=1)
            mask = mask.repeat((1, x.shape[1], 1, 1))
            x = x * mask

        if self.concat == 'input-discard_out1':
            x = x
        elif self.concat == 'input_softmax_out1':
            out1 = F.softmax(output1, dim=1)
            x = torch.cat([x, out1], dim=1)
        elif self.concat == 'input':
            x = torch.cat([x, output1], dim=1)
        elif self.concat == 'softmax_out1':
            x = F.softmax(output1, dim=1)
        else:
            x = output1

        if self.batch_norm:
            x = self.bn(x)

        output2 = self.unet2(x)
        return output1, output2
