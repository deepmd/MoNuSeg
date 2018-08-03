from .unet_parts import *
from .d_unet_model import DUNet
from .wired_unet_model import WiredUNet


class DWiredUNet(DUNet):
    def __init__(self, config):
        super(DWiredUNet, self).__init__(config)
        if len(config['unet1']['down']) != len(config['unet2']['down']):
            raise ValueError('Length of \'down\' of both UNets should be the same.')
        self.unet2 = WiredUNet(config['unet2'], config['unet1'])
        self.bn = nn.BatchNorm2d(config['unet2']['in_channels'])

    def forward(self, x):
        unet1_d_outs = []
        unet2_d_outs = []
        inp = x

        # UNet1
        for down in self.unet1.downs:
            x, before_pool = down(x)
            unet1_d_outs.append(before_pool)
        for base in self.unet1.bases:
            x = base(x)
        for up, d_out in zip(self.unet1.ups, reversed(unet1_d_outs)):
            x = up(x, d_out)
        output1 = self.unet1.outc(x)

        if self.masking is not None:
            out1 = F.softmax(output1, dim=1)
            mask = (out1.argmax(dim=1, keepdim=True) == self.mask_dim).float() if self.masking == 'hard' else \
                   out1[:, self.mask_dim]
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
