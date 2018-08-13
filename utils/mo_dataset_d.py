from common import *
from consts import *
from utils import helper
from utils import augmentation
from scipy.ndimage import zoom
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class MODatasetD(Dataset):
    """Multi Organ Dataset for Double UNet"""

    def __init__(self, root_dir, ids, num_patches=None, patch_size=None, transform=None, bgr=False,
                 masks=['inside', 'touching'], numeric_mask=True, centroid_size=5, image_scale=1,
                 scale_to_transform=1, get_areas=False):
        self.ids = ids
        self.transform = transform
        self.req_masks = masks
        self.numeric_mask = numeric_mask
        self.centroid_size = centroid_size
        self.scale = image_scale
        self.patch_coords = None
        self.transform_scale = scale_to_transform
        self.get_areas = get_areas
        if num_patches is not None and patch_size is not None:
            self.ids = np.random.permutation(np.repeat(ids, num_patches))
            patch_info_path = os.path.join(root_dir, 'patches-{:d}-{:d}.csv'.format(num_patches, patch_size))
            if os.path.isfile(patch_info_path):
                self.patch_coords = np.genfromtxt(patch_info_path, delimiter=',', dtype=np.int)
                self.patch_coords = self.patch_coords if self.patch_coords.ndim > 1 else np.expand_dims(self.patch_coords, axis=0)
            else:
                img_path = os.path.join(root_dir, IMAGES_DIR, self.ids[0]+'.tif')
                img = cv2.imread(img_path)
                self.patch_coords = np.zeros((len(self.ids), 4), dtype=np.int)
                self.patch_coords[:, 0] = np.random.randint(0, img.shape[0]-patch_size, (len(self.ids)))
                self.patch_coords[:, 1] = np.random.randint(0, img.shape[1]-patch_size, (len(self.ids)))
                self.patch_coords[:, 2] = self.patch_coords[:, 0] + patch_size
                self.patch_coords[:, 3] = self.patch_coords[:, 1] + patch_size
                np.savetxt(patch_info_path, self.patch_coords, delimiter=',', fmt='%d')
        self.images = {}
        self.masks = {}
        self.labels = {}
        for img_id in ids:
            img_path = os.path.join(root_dir, IMAGES_DIR, img_id+'.tif')
            img = cv2.imread(img_path)
            if image_scale > 1:
                img = cv2.resize(img, (img.shape[0]*image_scale, img.shape[1]*image_scale),
                                 interpolation=cv2.INTER_LANCZOS4)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not bgr else img
            self.images[img_id] = img
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
            labels_path = os.path.join(root_dir, LABELS_DIR, img_id + '.npy')
            labels = np.load(labels_path)
            self.labels[img_id] = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.images[self.ids[idx]]
        gt_masks = self.masks[self.ids[idx]]
        labels = self.labels[self.ids[idx]]

        if self.patch_coords is not None:
            y1, x1, y2, x2 = self.patch_coords[idx]
            img = img[y1*self.scale:y2*self.scale, x1*self.scale:x2*self.scale, :]
            masks = [gt_masks[m][y1:y2, x1:x2] for m in self.req_masks]
            labels = labels[y1:y2, x1:x2]
        else:
            masks = [gt_masks[m] for m in self.req_masks]

        H, W = img.shape[0]//self.scale, img.shape[1]//self.scale

        labels = [labels == label for label in np.unique(labels) if label != 0]
        labels = np.stack(labels, axis=-1).astype(np.uint8) if len(labels) > 0 else \
            np.zeros((H, W, 1), dtype=np.uint8)

        masks = np.stack(masks, axis=-1)
        if masks.ndim == 2:
            masks = np.expand_dims(masks, axis=-1)

        if self.transform is not None:
            patch_size = masks.shape[0]
            if self.transform_scale != 1:
                labels = cv2.resize(labels, (int(patch_size*self.transform_scale), int(patch_size*self.transform_scale)),
                                    interpolation=cv2.INTER_LINEAR)
                labels = (labels > 0).astype(np.uint8)
            img, masks, labels = self.transform(img, masks.astype(np.float), labels)
            if self.transform_scale != 1:
                labels = cv2.resize(labels, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                labels = (labels > 0).astype(np.uint8)

        masks = np.moveaxis(masks, -1, 0)
        if self.numeric_mask:
            n_mask = masks[0]
            for i in range(1, len(self.req_masks)):
                n_mask = np.maximum(n_mask, masks[i] * (i + 1))
            masks = np.expand_dims(n_mask, axis=0)

        if self.get_areas:
            centroids, areas = helper.get_centroids_areas(labels, centroid_size=self.centroid_size)
        else:
            centroids = helper.get_centroids(labels, centroid_size=self.centroid_size)
        centroids = np.expand_dims(centroids, axis=0)

        sample = {'image': img, 'masks': masks, 'centroids': centroids} if not self.get_areas else \
                 {'image': img, 'masks': masks, 'centroids': centroids, 'areas': areas}

        # DWM = helper.get_distance_transform_based_weight_map(centroids, BETA_IN_DISTANCE_WEIGHT)
        # sample = {'image': img, 'masks': masks, 'centroids': centroids, 'weight': DWM}

        return sample


# -----------------------------------------------------------------------
def train_transforms(image, masks, labels):
    seq = augmentation.get_train_augmenters_seq2(mode='constant')
    hooks_masks = augmentation.get_train_masks_augmenters_deactivator()

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
    image = seq_det.augment_images([image])[0]
    image = transforms.ToTensor()(image.copy())
    # image = transforms.Normalize(IMAGES_MEAN, IMAGES_STD)(image)

    masks = seq_det.augment_images([masks], hooks=hooks_masks)[0]
    masks = (masks >= MASK_THRESHOLD).astype(np.uint8)

    labels = seq_det.augment_images([labels], hooks=hooks_masks)[0]
    labels = (labels > 0).astype(np.uint8)
    # for index in range(labels.shape[-1]):
    #     labels[..., index] = (labels[..., index] > 0).astype(np.uint8)

    return image, masks, labels


def run_check_dataset(transform=None):
    ids = ['TCGA-18-5592-01Z-00-DX1']
    dataset = MODatasetD('../../MoNuSeg Training Data', ids, num_patches=10, patch_size=256, transform=transform)

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
        # plt.imshow(np.squeeze(sample['centroids']))
        #plt.imshow(np.squeeze(sample['weight']), cmap=in_cmap, alpha=0.5)

        # plt.imshow(img)
        # plt.imshow(np.squeeze(sample['masks'] == 1), cmap=in_cmap, alpha=0.5)
        plt.imshow(np.squeeze(sample['masks'] == 2))  # , cmap=bn_cmap, alpha=0.5)
        plt.imshow(np.squeeze(sample['masks'] == 3), cmap=in_cmap, alpha=0.5)
        plt.imshow(np.squeeze(sample['masks'] == 1), cmap=bn_cmap, alpha=0.5)
        # plt.imshow(np.squeeze(sample['centroids']), cmap=bn_cmap, alpha=0.5)
        plt.show()
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_check_dataset(train_transforms)
    print('\nsuccess!')
