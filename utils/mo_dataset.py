from common import *
from consts import *
from utils import augmentation

class MODataset(Dataset):
    """Multi Organ Dataset"""

    def __init__(self, root_dir, ids, num_patches=None, patch_size=None, transform=None):
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
        self.images = {}
        self.inside_masks = {}
        self.boundary_masks = {}
        for img_id in ids:
            img_path = os.path.join(self.root_dir, IMAGES_DIR, img_id+'.tif')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images[img_id] = img
            inside_mask_path = os.path.join(self.root_dir, INSIDE_MASKS_DIR, img_id+'.png')
            inside_mask = cv2.imread(inside_mask_path, cv2.IMREAD_GRAYSCALE) / 255
            self.inside_masks[img_id] = inside_mask
            boundary_mask_path = os.path.join(self.root_dir, BOUNDARY_MASKS_DIR, img_id+'.png')
            boundary_mask = cv2.imread(boundary_mask_path, cv2.IMREAD_GRAYSCALE) / 255
            self.boundary_masks[img_id] = boundary_mask

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.images[self.ids[idx]]
        inside_mask = self.inside_masks[self.ids[idx]]
        boundary_mask = self.boundary_masks[self.ids[idx]]

        if self.patch_coords is not None:
            y1, x1, y2, x2 = self.patch_coords[idx]
            img             = img[y1:y2, x1:x2, :]
            inside_mask     = inside_mask[y1:y2, x1:x2]
            boundary_mask   = boundary_mask[y1:y2, x1:x2]

        masks = np.stack([inside_mask, boundary_mask], axis=-1)
        if self.transform is not None:
            img, masks = self.transform(img, masks)

        background_mask = 1 - np.any(masks, axis=-1, keepdims=True)
        masks = np.append(masks, background_mask, axis=-1)
        masks = np.moveaxis(masks, -1, 0)
        sample = {'image': img, 'masks': masks}
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
        bn_cmap = colors.ListedColormap(bn_cmap)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.imshow(img)
        plt.imshow(sample['masks'][0], cmap=in_cmap, alpha=0.5)
        plt.imshow(sample['masks'][1], cmap=bn_cmap, alpha=0.5)
        plt.show()
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_check_dataset(train_transforms)
    print('\nsuccess!')
