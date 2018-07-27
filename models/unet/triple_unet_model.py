from .unet_parts import *
from ..unet import UNet
from ..res_unet import Res_UNet


class TripleUNet(nn.Module):
    def __init__(self, config):
        super(TripleUNet, self).__init__()
        config1 = config['unet1']
        config2 = config['unet2']
        config3 = config['unet3']
        self.concat = config.get('concat', None)
        config2['in_channels'] = config1['in_channels']
        config3['in_channels'] = config2['out_channels']
        if self.concat == 'input':
            config3['in_channels'] += config2['in_channels']
        elif self.concat == 'penultimate':
            config2['penultimate_output'] = True
            config3['in_channels'] += config2['up'][-1][0]
        self.unet1 = UNet(config1) if 'resnet' not in config1 else Res_UNet(config1['resnet'])
        self.unet2 = UNet(config2)
        self.unet3 = UNet(config3)
        self.l2_norm = normalize()
        self.bn = nn.BatchNorm2d(config['unet3']['in_channels'])

    def forward(self, x):
        output1 = self.unet1(x)
        mask = output1[:, 0]
        mask = torch.unsqueeze(mask, dim=1)
        mask_in = mask.repeat((1, x.shape[1], 1, 1))
        x = x * mask_in

        if self.concat == 'input':
            output2 = self.l2_norm(self.unet2(x))
            mask_out = mask.repeat((1, output2.shape[1], 1, 1))
            x = torch.cat([x, output2*mask_out], dim=1)
            x = self.bn(x)
        elif self.concat == 'penultimate':
            penultimate, output2 = self.unet2(x)
            output2 = self.l2_norm(output2)
            mask_out = mask.repeat((1, output2.shape[1], 1, 1))
            x = torch.cat([penultimate, output2*mask_out], dim=1)
            x = self.bn(x)
        else:
            output2 = self.l2_norm(self.unet2(x))
            mask_out = mask.repeat((1, output2.shape[1], 1, 1))
            x = output2 * mask_out

        output3 = self.unet3(x)
        return output1, output2, output3
