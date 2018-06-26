INPUT_DIR = '../MoNuSeg Training Data/'
WEIGHTS_DIR = './weights'
OUTPUT_DIR = './outputs'

IMAGES_DIR = 'Tissue images'
ANNOTATIONS_DIR = 'Annotations'

MASKS_DIR = 'Masks'
INSIDE_MASKS_DIR = 'Masks/Inside'
BOUNDARY_MASKS_DIR = 'Masks/Boundary'
LABELS_DIR = 'Labels'
COLORED_LABELS_DIR = 'Labels/Colored'
BOUNDARY_COLORED_LABELS_DIR = 'Labels/Boundary_Colored'

TEST_IDS = ['TCGA-18-5592-01Z-00-DX1', 'TCGA-38-6178-01Z-00-DX1', 'TCGA-A7-A13E-01Z-00-DX1', 'TCGA-A7-A13F-01Z-00-DX1',
            'TCGA-B0-5711-01Z-00-DX1', 'TCGA-G9-6336-01Z-00-DX1', 'TCGA-G9-6348-01Z-00-DX1', 'TCGA-HE-7128-01Z-00-DX1']

UNET_CONFIG = {'in_channels': 3, 'out_channels': 3,
               'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
               'base': [(512, 2)],
               'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
               'up_method': 'bilinear'}

DOUBLE_UNET_CONFIG = {
    'unet1': {'in_channels': 3, 'out_channels': 4,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'unet2': {'in_channels': 7, 'out_channels': 5,
              'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
              'base': [(512, 2)],
              'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
              'up_method': 'bilinear'},
    'concat_input': True
}
