from .unet_parts import *
from .double_unet_model import DoubleUNet
from .wired_unet_model import WiredUNet


class DoubleWiredUNet_GateInput(DoubleUNet):
    def __init__(self, config):
        config['unet1']['out_channels'] += config['unet1']['in_channels']
        super(DoubleWiredUNet_GateInput, self).__init__(config)
        if len(config['unet1']['down']) != len(config['unet2']['down']):
            raise ValueError('Length of \'down\' of both UNets should be the same.')
        config['unet2']['in_channels'] -= config['unet1']['in_channels']
        self.unet2 = WiredUNet(config['unet2'], config['unet1'])
        self.l2_norm = normalize()
        self.bn = nn.BatchNorm2d(config['unet2']['in_channels'])

    def forward(self, x):
        unet1_d_outs = []
        unet2_d_outs = []
        inp_channels = x.shape[1]

        # UNet1
        for down in self.unet1.downs:
            x, before_pool = down(x)
            unet1_d_outs.append(before_pool)
        for base in self.unet1.bases:
            x = base(x)
        for up, d_out in zip(self.unet1.ups, reversed(unet1_d_outs)):
            x = up(x, d_out)
        output1 = self.unet1.outc(x)
        output1[:, :-inp_channels] = self.l2_norm(output1[:, :-inp_channels])
        output1[:, -inp_channels:] = F.sigmoid(output1[:, -inp_channels:])

        x = self.bn(output1)

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

