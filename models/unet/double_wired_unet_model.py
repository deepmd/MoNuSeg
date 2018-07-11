from .unet_parts import *
from ..unet import UNet, DoubleUNet


class DoubleWiredUNet(DoubleUNet):
    def __init__(self, config):
        super(DoubleWiredUNet, self).__init__(config)
        if len(config['unet1']['down']) != len(config['unet2']['down']):
            raise ValueError('Length of \'down\' of both UNets should be the same.')
        self.unet2 = UNet2(config['unet2'], config['unet1'])

    def forward(self, x):
        unet1_d_outs = []
        unet2_d_outs = []
        inp = x if self.concat == 'input' else None

        # UNet1
        for down in self.unet1.downs:
            x, before_pool = down(x)
            unet1_d_outs.append(before_pool)
        for base in self.unet1.bases:
            x = base(x)
        for up, d_out in zip(self.unet1.ups, reversed(unet1_d_outs)):
            x = up(x, d_out)
        output1 = self.unet1.outc(x)

        if self.concat == 'input':
            x = torch.cat([inp, output1], dim=1)
        elif self.concat == 'penultimate':
            x = torch.cat([x, output1], dim=1)
        else:
            x = output1

        # UNet2
        for down, d_out1 in zip(self.unet2.downs, unet1_d_outs):
            x, before_pool = down(x, d_out1)
            unet2_d_outs.append(before_pool)
        for base in self.unet2.bases:
            x = base(x)
        for up, d_out in zip(self.unet2.ups, reversed(unet2_d_outs)):
            x = up(x, d_out)
        output2 = self.unet2.outc(x)

        return output1, output2


class UNet2(UNet):
    def __init__(self, config2, config1):
        super(UNet2, self).__init__(config2)

        downs = []
        add_se = config2.get('add_se', False)
        last_channels = config2['in_channels']
        for (d_channels2, n), (d_channels1, _) in zip(config2['down'], config1['down']):
            downs.append(down_merge(last_channels, d_channels1, d_channels2, n, add_se))
            last_channels = d_channels2
        self.downs = nn.ModuleList(downs)
