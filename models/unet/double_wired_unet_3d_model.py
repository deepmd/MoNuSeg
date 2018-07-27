from .unet_parts import *
from .double_unet_model import DoubleUNet
from .unet_model2 import UNet2


class DoubleWiredUNet_3d(DoubleUNet):
    def __init__(self, config):
        super(DoubleWiredUNet_3d, self).__init__(config)
        if len(config['unet1']['down']) != len(config['unet2']['down']):
            raise ValueError('Length of \'down\' of both UNets should be the same.')
        self.masking = config.get('masking', [])
        self.unet2 = UNet2(config['unet2'], config['unet1'])
        self.l2_norm = normalize()
        self.bn = nn.BatchNorm2d(config['unet2']['in_channels'])

    def forward(self, x):
        unet1_d_outs = []
        unet2_d_outs = []
        inp = x if self.concat == 'input' or self.concat == 'input-discard_out1' else None

        # UNet1
        for down in self.unet1.downs:
            x, before_pool = down(x)
            unet1_d_outs.append(before_pool)
        for base in self.unet1.bases:
            x = base(x)
        for up, d_out in zip(self.unet1.ups, reversed(unet1_d_outs)):
            x = up(x, d_out)
        output1 = self.unet1.outc(x)
        output1 = self.l2_norm(output1)

        mask_out = None
        mask_in = None
        if self.masking:
            mask = 1 - torch.gt((output1[:, 2] + output1[:, 5]), 1).float()
            # mask = 1 - ((output1[:, 2] + output1[:, 5]) / 2)
            mask = torch.unsqueeze(mask, dim=1)
            mask_out = mask.repeat((1, output1.shape[1], 1, 1)) if 'vector' in self.masking else None
            mask_in = mask.repeat((1, inp.shape[1], 1, 1)) if inp is not None and 'input' in self.masking else None

        if self.concat == 'input-discard_out1':
            x = (inp*mask_in if mask_in is not None else inp)
        elif self.concat == 'input':
            x = torch.cat([(inp*mask_in if mask_in is not None else inp), (output1*mask_out if mask_out is not None else output1)], dim=1)
            x = self.bn(x)
        elif self.concat == 'penultimate':
            x = torch.cat([x, (output1*mask_out if mask_out is not None else output1)], dim=1)
            x = self.bn(x)
        else:
            x = (output1*mask_out if mask_out is not None else output1)

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
