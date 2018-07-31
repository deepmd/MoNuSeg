from bokeh.core.query import IN

from consts import *
from os import path
from glob import glob
import cv2
from skimage import morphology as skmorph, io


mask_names = glob(path.join(INPUT_DIR, MASKS_DIR, '*.png'))
for mask_name in mask_names:
    gt_mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE) / 255
    mask = skmorph.binary_dilation(gt_mask, selem=skmorph.square(5))
    mask = skmorph.binary_erosion(mask, selem=skmorph.square(5))
    mask = gt_mask - mask
    mask = skmorph.binary_dilation(mask, selem=skmorph.square(5))

    save_path = path.join(INPUT_DIR, TOUCHING_BORDERS_DIR, mask_name.split('/')[-1])
    io.imsave(save_path, mask*255)
    print('{} saved.'.format(mask_name.split('/')[-1]))



