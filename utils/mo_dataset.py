from common import *
from consts import *
from utils import augmentation, helper

class MODataset(Dataset):
    """Multi Organ Dataset"""
    """Parameters
       ----------
       masks : List of values 'mask', 'inside', 'boundary', 'background', 'touching'
       inputs: List of values 'img', 'pred_mask'
    """

    def __init__(self, root_dir, ids, weights=None, num_patches=None, patch_size=None, transform=None, bgr=False,
                 inputs=['img'], masks=['inside', 'boundary', 'background'], numeric_mask=False):
        self.ids = ids
        self.transform = transform
        self.req_inputs = inputs
        self.req_masks = masks
        self.numeric_mask = numeric_mask
        self.patch_coords = None
        if num_patches is not None and patch_size is not None:
            repeats = [int(weights[id]*num_patches) for id in self.ids] if weights is not None else num_patches
            self.ids = np.random.permutation(np.repeat(ids, repeats))
            patch_info_path = os.path.join(root_dir, 'patches-{:d}-{:d}.csv'.format(num_patches, patch_size))
            if os.path.isfile(patch_info_path):
                self.patch_coords = np.genfromtxt(patch_info_path, delimiter=',', dtype=np.int)
                self.patch_coords = self.patch_coords if self.patch_coords.ndim > 1 else np.expand_dims(self.patch_coords, axis=0)
            else:
                img_path = os.path.join(root_dir, IMAGES_DIR, self.ids[0]+'.tif')
                img = cv2.imread(img_path)
                self.patch_coords = np.zeros((len(self.ids), 4), dtype=np.int)
                self.patch_coords[:, 0] = np.random.randint(0, img.shape[:-1][0]-patch_size, (len(self.ids)))
                self.patch_coords[:, 1] = np.random.randint(0, img.shape[:-1][1]-patch_size, (len(self.ids)))
                self.patch_coords[:, 2] = self.patch_coords[:, 0] + patch_size
                self.patch_coords[:, 3] = self.patch_coords[:, 1] + patch_size
                np.savetxt(patch_info_path, self.patch_coords, delimiter=',', fmt='%d')
        self.inputs = {}
        self.masks = {}
        for img_id in ids:
            self.inputs[img_id] = {}
            if 'img' in self.req_inputs:
                img_path = os.path.join(root_dir, IMAGES_DIR, img_id+'.tif')
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not bgr else img
                self.inputs[img_id]['img'] = img
            if 'pred_mask' in self.req_inputs:
                pred_mask_path = os.path.join(root_dir, PRED_MASKS_DIR, img_id+'.png')
                pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                self.inputs[img_id]['pred_mask'] = np.expand_dims(pred_mask, -1)

            self.masks[img_id] = {}
            if 'inside' in self.req_masks:
                inside_mask_path = os.path.join(root_dir, INSIDE_MASKS_DIR, img_id+'.png')
                inside_mask = cv2.imread(inside_mask_path, cv2.IMREAD_GRAYSCALE) / 255
                self.masks[img_id]['inside'] = inside_mask
            if 'boundary' in self.req_masks:
                boundary_mask_path = os.path.join(root_dir, BOUNDARY_MASKS_DIR, img_id+'.png')
                boundary_mask = cv2.imread(boundary_mask_path, cv2.IMREAD_GRAYSCALE) / 255
                self.masks[img_id]['boundary'] = boundary_mask
            if 'mask' in self.req_masks or 'background' in self.req_masks:
                mask_path = os.path.join(root_dir, MASKS_DIR, img_id+'.png')
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
                self.masks[img_id]['mask'] = mask
                self.masks[img_id]['background'] = 1 - mask
            if 'touching' in self.req_masks:
                touching_path = os.path.join(root_dir, TOUCHING_BORDERS_DIR, img_id+'.png')
                touching_border = cv2.imread(touching_path, cv2.IMREAD_GRAYSCALE) / 255
                self.masks[img_id]['touching'] = touching_border

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        inputs = self.inputs[self.ids[idx]]
        gt_masks = self.masks[self.ids[idx]]

        if self.patch_coords is not None:
            y1, x1, y2, x2 = self.patch_coords[idx]
            inputs = [inputs[n][y1:y2, x1:x2, :] for n in self.req_inputs]
            masks = [gt_masks[m][y1:y2, x1:x2] for m in self.req_masks]
        else:
            inputs = [inputs[n] for n in self.req_inputs]
            masks = [gt_masks[m] for m in self.req_masks]

        masks = np.stack(masks, axis=-1)
        inputs = np.concatenate(inputs, axis=-1)
        if masks.ndim == 2:
            masks = np.expand_dims(masks, axis=-1)

        if self.transform is not None:
            inputs, masks = self.transform(inputs, masks.astype(np.float))

        masks = np.moveaxis(masks, -1, 0)
        if self.numeric_mask:
            n_mask = masks[0]
            for i in range(1, len(self.req_masks)):
                n_mask = np.maximum(n_mask, masks[i]*(i+1))
            masks = np.expand_dims(n_mask, axis=0)

        # DWM = helper.get_distance_transform_based_weight_map(mask, beta=BETA_IN_DISTANCE_WEIGHT)
        # sample = {'image': img, 'masks': mask, 'weights': DWM}

        sample = {'image': inputs, 'masks': masks}
        return sample


# -----------------------------------------------------------------------
def train_transforms(image, masks):
    seq = augmentation.get_train_augmenters_seq1()
    hooks_masks = augmentation.get_train_masks_augmenters_deactivator()

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
    image_aug = seq_det.augment_images([image])[0]
    masks_aug = seq_det.augment_images([masks], hooks=hooks_masks)[0]

    image_aug_tensor = transforms.ToTensor()(image_aug.copy())
    masks_aug = (masks_aug >= MASK_THRESHOLD).astype(np.uint8)

    return image_aug_tensor, masks_aug


def run_check_dataset(transform=None):
    # ids = ['TCGA-18-5592-01Z-00-DX1']
    ids = ['TCGA-DK-A2I6-01A-01-TS1', 'TCGA-21-5786-01Z-00-DX1', 'TCGA-G2-A2EK-01A-02-TSB']
    dataset = MODataset('../../MoNuSeg Training Data', ids, weights=TRAIN_IDS_WEIGHTS,
                        num_patches=10, patch_size=256, transform=transform,
                        inputs=['img'], masks=['inside', 'touching'], numeric_mask=True)

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
        # plt.imshow(sample['masks'][0], cmap=in_cmap, alpha=0.5)
        # plt.imshow(sample['masks'][1], cmap=bn_cmap, alpha=0.5)
        plt.imshow(np.squeeze(sample['masks'] == 1), cmap=in_cmap, alpha=0.5)
        plt.imshow(np.squeeze(sample['masks'] == 2), cmap=bn_cmap, alpha=0.5)
        plt.show()
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_check_dataset(train_transforms)
    print('\nsuccess!')
