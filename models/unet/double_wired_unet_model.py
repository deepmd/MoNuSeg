from .unet_parts import *
from ..unet import UNet


class DoubleWiredUNet(nn.Module):
    def __init__(self, config):
        super(DoubleWiredUNet, self).__init__()
        if len(config['unet1']['down']) != len(config['unet2']['up']):
            raise ValueError('Length of \'down\' and \'up\' should be the same for both UNets.')
        self.concat_input = config.get('concat_input', False)
        self.unet1 = UNet(config['unet1'])
        self.unet2 = UNet2(config['unet2'], config['unet1'])

    def forward(self, x):
        unet1_d_outs = []
        unet2_d_outs = []
        inp = x if self.concat_input else None

        # UNet1
        for down in self.unet1.downs:
            x, before_pool = down(x)
            unet1_d_outs.append(before_pool)
        for base in self.unet1.bases:
            x = base(x)
        for up, d_out in zip(self.unet1.ups, reversed(unet1_d_outs)):
            x = up(x, d_out)
        output1 = self.unet1.outc(x)

        x = torch.cat([inp, output1], dim=1) if self.concat_input else torch.cat([x, output1], dim=1)

        # UNet2
        for down in self.unet2.downs:
            x, before_pool = down(x)
            unet2_d_outs.append(before_pool)
        for base in self.unet2.bases:
            x = base(x)
        for up, d_out1, d_out2 in zip(self.unet2.ups, reversed(unet1_d_outs), reversed(unet2_d_outs)):
            d_out = torch.cat([d_out2, d_out1], dim=1)
            x = up(x, d_out)
        output2 = self.unet2.outc(x)

        return output1, output2


class UNet2(UNet):
    def __init__(self, config2, config1):
        super(UNet2, self).__init__(config2)

        ups = []
        last_channels = config2['base'][-1][0] if config2['base'] else config2['down'][-1][0]
        bilinear_up = config2['up_method'] == 'bilinear'
        for (u_channels, n), (d_channels2, _), (d_channels1, _) in zip(config2['up'], reversed(config2['down']), reversed(config1['down'])):
            ups.append(up(last_channels + d_channels2 + d_channels1, u_channels, n, bilinear_up))
            last_channels = u_channels
        self.ups = nn.ModuleList(ups)
