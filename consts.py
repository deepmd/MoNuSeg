INPUT_DIR = '../MoNuSeg Training Data/'
WEIGHTS_DIR = './weights'
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

TRAIN_IDS_WEIGHTS = None

MASK_THRESHOLD = 0.5
DEACTIVATED_MASK_AUG_LIST = ['Superpixels', 'GaussianBlur', 'AverageBlur', 'MedianBlur', 'Sharpen', 'Emboss',
                             'EdgeDetect', 'DirectedEdgeDetect', 'AdditiveGaussianNoise', 'Dropout',
                             'CoarseDropout', 'Invert', 'Add_Value_to_each_Pixel', 'Change_Brightness',
                             'ContrastNormalization', 'Grayscale']

# BETA_IN_DISTANCE_WEIGHT = 20

IMAGES_MEAN = [0.8275685641750257, 0.5215321518722066, 0.646311050624383, 0.5]
IMAGES_STD = [0.16204139441725898, 0.248547854527502, 0.2014914668413328, 1]


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