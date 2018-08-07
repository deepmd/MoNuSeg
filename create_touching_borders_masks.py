from common import *
from consts import *
from skimage import io
import skimage.morphology as skmorph
from scipy.ndimage import morphology


path = os.path.join(INPUT_DIR, TOUCHING_BORDERS_DIR, 'test')
if not os.path.exists(path):
    os.makedirs(path)

mask_names = glob.glob(os.path.join(INPUT_DIR, INSIDE_MASKS_DIR, '*.png'))
for mask_name in mask_names:
    gt_mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE) / 255
    mask = skmorph.binary_dilation(gt_mask, selem=skmorph.square(5))
    mask = skmorph.binary_erosion(mask, selem=skmorph.square(5))
    mask = gt_mask - mask
    mask = skmorph.binary_dilation(mask, selem=skmorph.square(7))
    mask = skmorph.binary_erosion(mask, selem=skmorph.square(6))
    mask = skmorph.remove_small_objects(mask, min_size=30)

    save_path = os.path.join(path, mask_name.split('/')[-1])
    io.imsave(save_path, mask*255)
    print('{} saved.'.format(mask_name.split('/')[-1]))



