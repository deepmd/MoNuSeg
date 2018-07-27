from .unet_parts import *
from .triple_unet_model import TripleUNet
from .wired_unet_model import WiredUNet


class TripleWiredUNet(TripleUNet):
    def __init__(self, config):
        super(TripleWiredUNet, self).__init__(config)
        if len(config['unet2']['down']) != len(config['unet3']['down']):
            raise ValueError('Length of \'down\' of UNets 2 and 3 should be the same.')
        self.unet3 = WiredUNet(config['unet3'], config['unet2'])
        self.l2_norm = normalize()
        self.bn = nn.BatchNorm2d(config['unet3']['in_channels'])

    def forward(self, x):
        unet2_d_outs = []
        unet3_d_outs = []
        output1 = self.unet1(x)

        mask = output1[:, 0]
        mask = torch.unsqueeze(mask, dim=1)
        mask_in = mask.repeat((1, x.shape[1], 1, 1))
        x = x * mask_in
        inp = x if self.concat == 'input' else None

        # UNet2
        for down in self.unet2.downs:
            x, before_pool = down(x)
            unet2_d_outs.append(before_pool)
        for base in self.unet2.bases:
            x = base(x)
        for up, d_out in zip(self.unet2.ups, reversed(unet2_d_outs)):
            x = up(x, d_out)
        output2 = self.unet2.outc(x)
        output2 = self.l2_norm(output2)
        mask_out = mask.repeat((1, output2.shape[1], 1, 1))

        if self.concat == 'input':
            x = torch.cat([inp, output2*mask_out], dim=1)
            x = self.bn(x)
        elif self.concat == 'penultimate':
            x = torch.cat([x, output2*mask_out], dim=1)
            x = self.bn(x)
        else:
            x = output2 * mask_out

        # UNet3
        for down, d_out1 in zip(self.unet3.downs, unet2_d_outs):
            x, before_pool = down(x, d_out1)
            unet3_d_outs.append(before_pool)
        for base in self.unet3.bases:
            x = base(x)
        for up, d_out in zip(self.unet3.ups, reversed(unet3_d_outs)):
            x = up(x, d_out)
        output3 = self.unet3.outc(x)

        return output1, output2, output3
