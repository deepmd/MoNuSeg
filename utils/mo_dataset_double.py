from common import *
from consts import *
from utils.mo_dataset import MODataset
from utils import helper
from utils import augmentation
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class MODatasetDouble(MODataset):
    """Multi Organ Dataset for Double UNet"""

    def __init__(self, root_dir, ids, num_patches=None, patch_size=None, transform=None):
        super(MODatasetDouble, self).__init__(root_dir, ids, num_patches, patch_size, transform)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, IMAGES_DIR, self.ids[idx]+'.tif')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root_dir, MASKS_DIR, self.ids[idx]+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
        labels_path = os.path.join(self.root_dir, LABELS_DIR, self.ids[idx]+'.npy')
        labels = np.load(labels_path)
        # mask_1 = skmorph.binary_erosion(mask, skmorph.disk(1))
        # mask_3 = skmorph.binary_erosion(mask, skmorph.disk(3))
        # mask_5 = skmorph.binary_erosion(mask, skmorph.disk(5))

        if self.patch_coords is not None:
            y1, x1, y2, x2 = self.patch_coords[idx]
            img = img[y1:y2, x1:x2, :]
            mask = mask[y1:y2, x1:x2]
            # mask_1 = mask_1[y1:y2, x1:x2]
            # mask_3 = mask_3[y1:y2, x1:x2]
            # mask_5 = mask_5[y1:y2, x1:x2]
            labels = labels[y1:y2, x1:x2]

        labels, _, _ = skimage.segmentation.relabel_sequential(labels)
        labels = [(labels == label) for label in range(1, len(np.unique(labels)))]
        labels = np.stack(labels, axis=-1).astype(np.uint8)
        if self.transform is not None:
            img, mask, labels = self.transform(img, mask, labels)

        centroids, vectors, areas = helper.get_centroids_vectors_areas(labels, centroid_size=3)
        masks = np.stack([mask, centroids], axis=0)
        sample = {'image': img, 'masks': masks, 'vectors': vectors, 'areas': areas}

        return sample


# -----------------------------------------------------------------------
def train_transforms(image, mask, labels):
    seq = augmentation.get_train_augmenters_seq()
    hooks_masks = augmentation.get_train_masks_augmenters_deactivator()

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
    image_aug = seq_det.augment_images([image])[0]
    mask_aug = seq_det.augment_images([mask], hooks=hooks_masks)[0]
    labels_aug = seq_det.augment_images([labels], hooks=hooks_masks)[0]

    mask_aug = (mask_aug >= MASK_THRESHOLD).astype(np.uint8)
    for index in range(labels_aug.shape[-1]):
        labels_aug[..., index] = (labels_aug[..., index] > 0).astype(np.uint8)

    # merge_labels = np.max(masks_aug[:, :, 1:], axis=-1, keepdims=False)
    # merge_labels, _, _ = segmentation.relabel_sequential(merge_labels)
    # mask_labels_aug = np.stack([masks_aug[:, :, 0], merge_labels], axis=-1)

    image_aug_tensor = transforms.ToTensor()(image_aug.copy())
    # image_aug_tensor = transforms.Normalize([0.03072981, 0.03072981, 0.01682784],
    #                              [0.17293351, 0.12542403, 0.0771413 ])(image_aug_tensor)

    return image_aug_tensor, mask_aug, labels_aug


def run_check_dataset(transform=None):
    ids = ['TCGA-18-5592-01Z-00-DX1']
    dataset = MODatasetDouble('../../MoNuSeg Training Data', ids, num_patches=10, patch_size=256, transform=transform)

    for n in range(len(dataset)):
        sample = dataset[n]
        img = sample['image'] if transform is None else np.moveaxis(sample['image'].numpy(), 0, -1)
        in_cmap = colors.ListedColormap(['black', '#7CFC00'])
        in_cmap = in_cmap(np.arange(2))
        in_cmap[:, -1] = np.linspace(0, 1, 2)
        in_cmap = colors.ListedColormap(in_cmap)
        bn_cmap = colors.ListedColormap(['black', '#FF0000'])
        bn_cmap = bn_cmap(np.arange(2))
        bn_cmap[:, -1] = np.linspace(0, 1, 2)
        bn_cmap = colors.ListedColormap(bn_cmap)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.imshow(img)
        plt.imshow(sample['masks'][0], cmap=in_cmap, alpha=0.5)
        plt.imshow(sample['masks'][-1], cmap=bn_cmap, alpha=0.5)
        plt.show()
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_check_dataset(train_transforms)
    print('\nsuccess!')
