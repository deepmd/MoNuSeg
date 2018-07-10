from common import *
from consts import *
from utils import helper
from utils import augmentation
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class MODatasetDouble(Dataset):
    """Multi Organ Dataset for Double UNet"""

    def __init__(self, root_dir, ids, num_patches=None, patch_size=None, transform=None, bgr=False):
        self.root_dir = root_dir
        self.ids = ids
        self.transform = transform
        self.patch_coords = None
        if num_patches is not None and patch_size is not None:
            self.ids = np.random.permutation(np.repeat(ids, num_patches))
            patch_info_path = os.path.join(root_dir, 'patches-{:d}-{:d}.csv'.format(num_patches, patch_size))
            if os.path.isfile(patch_info_path):
                self.patch_coords = np.genfromtxt(patch_info_path, delimiter=',', dtype=np.int)
                self.patch_coords = self.patch_coords if self.patch_coords.ndim > 1 else np.expand_dims(
                    self.patch_coords, axis=0)
            else:
                img_path = os.path.join(self.root_dir, IMAGES_DIR, self.ids[0]+'.tif')
                img = cv2.imread(img_path)
                self.patch_coords = np.zeros((len(self.ids), 4), dtype=np.int)
                self.patch_coords[:, 0] = np.random.randint(0, img.shape[:-1][0] - patch_size, (len(self.ids)))
                self.patch_coords[:, 1] = np.random.randint(0, img.shape[:-1][1] - patch_size, (len(self.ids)))
                self.patch_coords[:, 2] = self.patch_coords[:, 0] + patch_size
                self.patch_coords[:, 3] = self.patch_coords[:, 1] + patch_size
                np.savetxt(patch_info_path, self.patch_coords, delimiter=',', fmt='%d')
        self.images = {}
        self.masks = {}
        self.labels = {}
        for img_id in ids:
            img_path = os.path.join(self.root_dir, IMAGES_DIR, img_id+'.tif')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not bgr else img
            self.images[img_id] = img
            mask_path = os.path.join(self.root_dir, MASKS_DIR, img_id+'.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
            self.masks[img_id] = mask
            labels_path = os.path.join(self.root_dir, LABELS_DIR, img_id+'.npy')
            labels = np.load(labels_path)
            self.labels[img_id] = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.images[self.ids[idx]]
        mask = self.masks[self.ids[idx]]
        labels = self.labels[self.ids[idx]]
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


def run_check_vectors(transform=None, num_vectors=10):
    ids = ['TCGA-18-5592-01Z-00-DX1']
    dataset = MODatasetDouble('../../MoNuSeg Training Data', ids, num_patches=10, patch_size=256, transform=transform)

    for n in range(len(dataset)):
        sample = dataset[n]
        (x_nonzeros, y_nonzeros) = np.nonzero(sample['masks'][0])
        random_indices = np.random.randint(0, len(x_nonzeros)-1, num_vectors)
        (x_origins, y_origins) = x_nonzeros[random_indices], y_nonzeros[random_indices]

        plt.quiver(*(y_origins, x_origins),
                   -sample['vectors'][1, x_origins, y_origins],
                   sample['vectors'][0, x_origins, y_origins],
                   color=['b'])

        plt.quiver(*(y_origins, x_origins),
                   -sample['vectors'][3, x_origins, y_origins],
                   sample['vectors'][2, x_origins, y_origins],
                   color=['r'])

        in_cmap = colors.ListedColormap(['black', '#7CFC00'])
        in_cmap = in_cmap(np.arange(2))
        in_cmap[:, -1] = np.linspace(0, 1, 2)
        in_cmap = colors.ListedColormap(in_cmap)
        bn_cmap = colors.ListedColormap(['black', '#FF0000'])
        bn_cmap = bn_cmap(np.arange(2))
        bn_cmap[:, -1] = np.linspace(0, 1, 2)
        bn_cmap = colors.ListedColormap(bn_cmap)
        plt.rcParams['axes.facecolor'] = 'black'
        # plt.imshow(img)
        plt.imshow(sample['masks'][0], cmap=in_cmap, alpha=0.5)
        plt.imshow(sample['masks'][-1], cmap=bn_cmap, alpha=0.5)
        plt.show()
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    # run_check_dataset(train_transforms)
    run_check_vectors(train_transforms, 50)
    print('\nsuccess!')
