from common import *
from consts import *
import imgaug as ia
from imgaug import augmenters as iaa


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


def get_train_augmenters_seq():
    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-90, 90),
                shear=(-16, 16),
                order=[0, 1],
                cval=0,
                mode='edge'
            )),

            # In some images distort local areas with varying strength.
            sometimes(iaa.PiecewiseAffine(
                scale=(0.01, 0.1),
                order=[0, 1],
                cval=0,
                mode='edge'
            )),

            # Blur some images with varying strength using
            # gaussian blur (sigma between 0 and 2.0),
            # average/uniform blur (kernel size between 2x2 and 5x5)
            # median blur (kernel size between 3x3 and 7x7).
            sometimes(iaa.OneOf([
                iaa.GaussianBlur((0, 2.0), name='GaussianBlur'),
                iaa.AverageBlur(k=(2, 5), name='AverageBlur'),
                iaa.MedianBlur(k=(3, 7), name='MedianBlur'),
            ]))
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    return seq


# change the activated augmenters for masks,
# we only want to execute for example horizontal flip, affine transformation and one of
# the gaussian noises
def train_activator_masks(images, augmenter, parents, default):
    if augmenter.name in DEACTIVATED_MASK_AUG_LIST:
        return False
    else:
        # default value for all other augmenters
        return default


def get_train_masks_augmenters_deactivator():
    return ia.HooksImages(activator=train_activator_masks)