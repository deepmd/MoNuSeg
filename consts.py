INPUT_DIR = '../MoNuSeg Training Data/'
WEIGHTS_DIR = './weights'
OUTPUT_DIR = './outputs'

IMAGES_DIR = 'Tissue images'
ANNOTATIONS_DIR = 'Annotations'
MASKS_DIR = 'Masks'
LABELS_DIR = 'Labels'

TEST_IDS = ['TCGA-18-5592-01Z-00-DX1', 'TCGA-38-6178-01Z-00-DX1', 'TCGA-A7-A13E-01Z-00-DX1', 'TCGA-A7-A13F-01Z-00-DX1',
            'TCGA-B0-5711-01Z-00-DX1', 'TCGA-G9-6336-01Z-00-DX1', 'TCGA-G9-6348-01Z-00-DX1', 'TCGA-HE-7128-01Z-00-DX1']

INSIDE_VALUE = 127
BOUNDARY_VALUE = 255

UNET_CONFIG = {'down': [(64, 2), (128, 2), (256, 2), (512, 2)],
               'base': [(512, 2)],
               'up'  : [(256, 2), (128, 2), (64, 2), (64, 2)],
               'up_method': 'bilinear'}
