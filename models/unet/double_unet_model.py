from common import *
from ..unet import UNet


class DoubleUNet(nn.Module):
    def __init__(self, config):
        super(DoubleUNet, self).__init__()
        config1 = config['unet1']
        config2 = config['unet2']
        self.concat = config.get('concat', None)
        config2['in_channels'] = config1['out_channels']
        if self.concat == 'input':
            config2['in_channels'] += config1['in_channels']
        elif self.concat == 'penultimate':
            config1['penultimate_output'] = True
            config2['in_channels'] += config1['up'][-1][0]
        self.unet1 = UNet(config1)
        self.unet2 = UNet(config2)

    def forward(self, x):
        if self.concat == 'input':
            output1 = self.unet1(x)
            x = torch.cat([x, output1], dim=1)
        elif self.concat == 'penultimate':
            penultimate, output1 = self.unet1(x)
            x = torch.cat([penultimate, output1], dim=1)
        else:
            x = output1 = self.unet1(x)
        output2 = self.unet2(x)
        return output1, output2