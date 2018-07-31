from common import *
from consts import *
from utils import augmentation, helper

class MODataset(Dataset):
    """Multi Organ Dataset"""

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
                self.patch_coords = self.patch_coords if self.patch_coords.ndim > 1 else np.expand_dims(self.patch_coords, axis=0)
            else:
                img_path = os.path.join(self.root_dir, IMAGES_DIR, self.ids[0]+'.tif')
                img = cv2.imread(img_path)
                self.patch_coords = np.zeros((len(self.ids), 4), dtype=np.int)
                self.patch_coords[:, 0] = np.random.randint(0, img.shape[:-1][0]-patch_size, (len(self.ids)))
                self.patch_coords[:, 1] = np.random.randint(0, img.shape[:-1][1]-patch_size, (len(self.ids)))
                self.patch_coords[:, 2] = self.patch_coords[:, 0] + patch_size
                self.patch_coords[:, 3] = self.patch_coords[:, 1] + patch_size
                np.savetxt(patch_info_path, self.patch_coords, delimiter=',', fmt='%d')
        self.bgr = bgr
        # self.images = {}
        # self.masks = {}
        # for img_id in ids:
        #     img_path = os.path.join(self.root_dir, IMAGES_DIR, img_id + '.tif')
        #     img = cv2.imread(img_path)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not bgr else img
        #     self.images[img_id] = img
        #     mask_path = os.path.join(self.root_dir, MASKS_DIR, img_id + '.png')
        #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
        #     self.masks[img_id] = mask

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # img = self.images[self.ids[idx]]
        # mask = self.masks[self.ids[idx]]
        img_path = os.path.join(self.root_dir, IMAGES_DIR, self.ids[idx] + '.tif')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not self.bgr else img
        mask_path = os.path.join(self.root_dir, MASKS_DIR, self.ids[idx] + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255

        if self.patch_coords is not None:
            y1, x1, y2, x2 = self.patch_coords[idx]
            img = img[y1:y2, x1:x2, :]
            mask = mask[y1:y2, x1:x2]

        mask = np.expand_dims(mask, axis=-1)
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        mask = np.moveaxis(mask, -1, 0)
        # DWM = helper.get_distance_transform_based_weight_map(mask, beta=BETA_IN_DISTANCE_WEIGHT)
        # sample = {'image': img, 'masks': mask, 'weights': DWM}

        sample = {'image': img, 'masks': mask}

        return sample


# -----------------------------------------------------------------------
def train_transforms(image, masks):
    seq = augmentation.get_train_augmenters_seq()
    hooks_masks = augmentation.get_train_masks_augmenters_deactivator()

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
    image_aug = seq_det.augment_images([image])[0]
    masks_aug = seq_det.augment_images([masks], hooks=hooks_masks)[0]

    image_aug_tensor = transforms.ToTensor()(image_aug.copy())
    # image_aug_tensor = transforms.Normalize([0.03072981, 0.03072981, 0.01682784],
    #                              [0.17293351, 0.12542403, 0.0771413 ])(image_aug_tensor)

    masks_aug = (masks_aug >= MASK_THRESHOLD).astype(np.uint8)

    return image_aug_tensor, masks_aug


def run_check_dataset(transform=None):
    ids = ['TCGA-18-5592-01Z-00-DX1']
    dataset = MODataset('../../MoNuSeg Training Data', ids, num_patches=10, patch_size=256, transform=transform)

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
        # bn_cmap = colors.ListedColormap(bn_cmap)
        plt.rcParams['axes.facecolor'] = 'black'
        # plt.imshow(np.squeeze(sample['weights']))
        plt.imshow(img)
        plt.imshow(np.squeeze(sample['masks']), cmap=in_cmap, alpha=0.5)
        # plt.imshow(sample['masks'][0], cmap=in_cmap, alpha=0.5)
        # plt.imshow(sample['masks'][1], cmap=bn_cmap, alpha=0.5)
        plt.show()
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_check_dataset(train_transforms)
    print('\nsuccess!')
