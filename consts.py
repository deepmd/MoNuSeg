INPUT_DIR = '../MoNuSeg Training Data/'
WEIGHTS_DIR = './weights'
SNAPSHOT_DIR = './weights/snapshots'
OUTPUT_DIR = './outputs'

IMAGES_DIR = 'Tissue images'
ANNOTATIONS_DIR = 'Annotations'

MASKS_DIR = 'Masks'
PRED_MASKS_DIR = 'Masks/Predicted_Masks'
INSIDE_MASKS_DIR = 'Masks/Inside'
BOUNDARY_MASKS_DIR = 'Masks/Boundary'
LABELS_DIR = 'Labels'
COLORED_LABELS_DIR = 'Labels/Colored'
BOUNDARY_COLORED_LABELS_DIR = 'Labels/Boundary_Colored'
TOUCHING_BORDERS_DIR = 'Masks/Touching_Borders'

TEST_IDS = ['TCGA-18-5592-01Z-00-DX1', 'TCGA-38-6178-01Z-00-DX1', 'TCGA-A7-A13E-01Z-00-DX1', 'TCGA-A7-A13F-01Z-00-DX1',
            'TCGA-B0-5711-01Z-00-DX1', 'TCGA-G9-6336-01Z-00-DX1', 'TCGA-G9-6348-01Z-00-DX1', 'TCGA-HE-7128-01Z-00-DX1']

D_TRAIN_IDS = ['TCGA-38-6178-01Z-00-DX1','TCGA-21-5784-01Z-00-DX1','TCGA-50-5931-01Z-00-DX1','TCGA-G2-A2EK-01A-02-TSB',
               'TCGA-A7-A13F-01Z-00-DX1','TCGA-21-5786-01Z-00-DX1','TCGA-AY-A8YK-01A-01-TS1','TCGA-49-4488-01Z-00-DX1',
               'TCGA-NH-A8F7-01A-01-TS1','TCGA-A7-A13E-01Z-00-DX1','TCGA-B0-5698-01Z-00-DX1','TCGA-HE-7130-01Z-00-DX1',
               'TCGA-B0-5711-01Z-00-DX1','TCGA-HE-7129-01Z-00-DX1','TCGA-E2-A1B5-01Z-00-DX1','TCGA-AR-A1AK-01Z-00-DX1',
               'TCGA-E2-A14V-01Z-00-DX1','TCGA-B0-5710-01Z-00-DX1','TCGA-AR-A1AS-01Z-00-DX1','TCGA-G9-6363-01Z-00-DX1',
               'TCGA-DK-A2I6-01A-01-TS1','TCGA-G9-6356-01Z-00-DX1','TCGA-KB-A93J-01A-01-TS1','TCGA-RD-A8N9-01A-01-TS1']
D_VALID_IDS = ['TCGA-CH-5767-01Z-00-DX1','TCGA-18-5592-01Z-00-DX1','TCGA-G9-6336-01Z-00-DX1','TCGA-HE-7128-01Z-00-DX1',
               'TCGA-G9-6362-01Z-00-DX1','TCGA-G9-6348-01Z-00-DX1']

R_TRAIN_IDS = ['TCGA-38-6178-01Z-00-DX1','TCGA-21-5784-01Z-00-DX1','TCGA-50-5931-01Z-00-DX1','TCGA-G2-A2EK-01A-02-TSB',
               'TCGA-A7-A13F-01Z-00-DX1','TCGA-21-5786-01Z-00-DX1','TCGA-AY-A8YK-01A-01-TS1','TCGA-49-4488-01Z-00-DX1',
               'TCGA-NH-A8F7-01A-01-TS1','TCGA-A7-A13E-01Z-00-DX1','TCGA-B0-5698-01Z-00-DX1','TCGA-HE-7130-01Z-00-DX1',
               'TCGA-B0-5711-01Z-00-DX1','TCGA-CH-5767-01Z-00-DX1','TCGA-18-5592-01Z-00-DX1','TCGA-HE-7129-01Z-00-DX1',
               'TCGA-E2-A1B5-01Z-00-DX1','TCGA-AR-A1AK-01Z-00-DX1','TCGA-G9-6336-01Z-00-DX1','TCGA-HE-7128-01Z-00-DX1',
               'TCGA-G9-6362-01Z-00-DX1','TCGA-E2-A14V-01Z-00-DX1','TCGA-G9-6348-01Z-00-DX1','TCGA-B0-5710-01Z-00-DX1']
R_VALID_IDS = ['TCGA-AR-A1AS-01Z-00-DX1','TCGA-G9-6363-01Z-00-DX1','TCGA-DK-A2I6-01A-01-TS1','TCGA-G9-6356-01Z-00-DX1',
               'TCGA-KB-A93J-01A-01-TS1','TCGA-RD-A8N9-01A-01-TS1']

TRAIN_IDS_WEIGHTS = None

MASK_THRESHOLD = 0.5
DEACTIVATED_MASK_AUG_LIST = ['Superpixels', 'GaussianBlur', 'AverageBlur', 'MedianBlur', 'Sharpen', 'Emboss',
                             'EdgeDetect', 'DirectedEdgeDetect', 'AdditiveGaussianNoise', 'Dropout',
                             'CoarseDropout', 'Invert', 'Add_Value_to_each_Pixel', 'Change_Brightness',
                             'ContrastNormalization', 'Grayscale']

BETA_IN_DISTANCE_WEIGHT = 40

IMAGES_MEAN = [0.8275685641750257, 0.5215321518722066, 0.646311050624383]
IMAGES_STD = [0.16204139441725898, 0.248547854527502, 0.2014914668413328]


# ------------------ SINGLE UNET CONFIG ----------------------------
UNET_CONFIG_0 = {'in_channels': 3, 'out_channels': 3,
               'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
               'base': [(512, 2)],
               'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
               'up_method': 'nearest'}

UNET_CONFIG_1 = {'in_channels': 3, 'out_channels': 1,
                 'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
                 'base': [(512, 2)],
                 'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
                 'up_method': 'bilinear'}

# ------------------ DOUBLE UNET CONFIG ----------------------------
DOUBLE_UNET_CONFIG_1 = {
    'unet1': {'in_channels': 3, 'out_channels': 4,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'input'
}

DOUBLE_UNET_CONFIG_2 = {
    'unet1': {'in_channels': 3, 'out_channels': 4,
              'down': [(64, 2), (96, 2), (128, 2), (192, 2), (256, 2), (384, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (192, 2), (128, 2), (96, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'input'
}

DOUBLE_UNET_CONFIG_3 = {
    'unet1': {'in_channels': 3, 'out_channels': 4,
              'down': [(64, 2), (96, 2), (128, 2), (192, 2), (256, 2), (384, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (192, 2), (128, 2), (96, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'penultimate'
}

DOUBLE_UNET_CONFIG_4 = {
    'unet1': {'in_channels': 3, 'out_channels': 4,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'penultimate'
}

DOUBLE_UNET_CONFIG_5 = {
    'unet1': {'in_channels': 3, 'out_channels': 4,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest',
              'add_se': True},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest',
              'add_se': True},
    'concat': 'input'
}

DOUBLE_UNET_CONFIG_6 = {
    'unet1': {'in_channels': 3, 'out_channels': 4,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'input-discard_out1'
}

# ------------------ DOUBLE UNET 3D CONFIG ----------------------------
DOUBLE_UNET_3D_CONFIG_1 = {
    'unet1': {'in_channels': 3, 'out_channels': 6,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'input',
    'masking': ['input', 'vector']
}

DOUBLE_UNET_3D_CONFIG_2 = {
    'unet1': {'in_channels': 3, 'out_channels': 6,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'input',
    'masking': ['input']
}

DOUBLE_UNET_3D_CONFIG_3 = {
    'unet1': {'in_channels': 3, 'out_channels': 6,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'input'
}

DOUBLE_UNET_3D_CONFIG_4 = {
    'unet1': {'in_channels': 3, 'out_channels': 6,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'penultimate'
}

# ------------------ TRIPLE UNET CONFIG ----------------------------
TRIPLE_UNET_CONFIG_1 = {
    'unet1': {'in_channels': 3, 'out_channels': 1,
              'down': [(64, 2), (96, 2), (128, 2), (192, 2), (256, 2), (384, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (192, 2), (128, 2), (96, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet2': {'out_channels': 4,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'unet3': {'out_channels': 2,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'nearest'},
    'concat': 'input'
}

# ------------------ D UNET CONFIG ----------------------------
D_UNET_CONFIG_1 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'unet2': {'out_channels': 1,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat': 'input'
}

D_UNET_CONFIG_2 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'unet2': {'out_channels': 1,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat': 'input-softmax_out1',
    'masking': 'hard', 'mask_dim': 1
}

D_UNET_CONFIG_3 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'unet2': {'out_channels': 1,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat': 'input-discard_out1',
    'masking': 'hard', 'mask_dim': 1
}

D_UNET_CONFIG_4 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'vgg': 16,
              'pretrained': True},
    'unet2': {'out_channels': 1,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat': 'input-discard_out1',
    'masking': 'hard', 'mask_dim': 1
}

D_UNET_CONFIG_5 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'vgg': 16,
              'pretrained': True},
    'unet2': {'out_channels': 1,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat': 'input-discard_out1',
    'masking': 'soft', 'mask_dim': 1
}

D_UNET_CONFIG_6 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'vgg': 16,
              'pretrained': True},
    'unet2': {'out_channels': 1,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat': 'input-softmax_out1',
    'masking': 'soft', 'mask_dim': 1,
    'batch_norm': True
}

D_UNET_CONFIG_7 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'vgg': 16,
              'pretrained': True},
    'unet2': {'out_channels': 1,
              'vgg': 16,
              'pretrained': True},
    'concat': 'input-discard_out1',
    'masking': 'soft', 'mask_dim': 1
}

D_UNET_CONFIG_8 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'unet2': {'out_channels': 1,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat': 'input-discard_out1',
    'masking': 'soft', 'mask_dim': 1
}

D_UNET_CONFIG_9 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'unet2': {'out_channels': 1,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat': 'input-discard_out1',
    'masking': 'soft', 'mask_dim': 1,
    'dense': True
}

D_UNET_CONFIG_10 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'resnet': 101,
              'pretrained': True},
    'unet2': {'out_channels': 1,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat': 'input-discard_out1',
    'masking': 'soft', 'mask_dim': 1
}

D_UNET_CONFIG_11 = {
    'unet1': {'in_channels': 3, 'out_channels': 3,
              'resnet': 101,
              'pretrained': True},
    'unet2': {'out_channels': 1,
              'resnet': 101,
              'pretrained': True},
    'concat': 'input-discard_out1',
    'masking': 'soft', 'mask_dim': 1
}

# ------------------ SAMS-NET CONFIG ----------------------------
SAMS_NET_CONFIG_1 = {
    'in_channels': 3, 'out_channels': 4,
    'down': [(64, 2), (128, 2), (256, 2)],
    'base': [(256, 2)],
    'up'  : [(256, 2), (128, 2), (64, 2)],
    'up_method': 'deconv'
}

SAMS_NET_CONFIG_2 = {
    'in_channels': 3, 'out_channels': 4,
    'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
    'base': [(512, 2)],
    'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
    'up_method': 'deconv'
}