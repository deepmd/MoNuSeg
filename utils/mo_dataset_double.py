from common import *
from consts import *
from utils.mo_dataset import MODataset
from utils import helper


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
        mask_1 = skmorph.binary_erosion(mask, skmorph.disk(2))
        mask_3 = skmorph.binary_erosion(mask, skmorph.disk(4))
        mask_5 = skmorph.binary_erosion(mask, skmorph.disk(6))

        if self.patch_coords is not None:
            y1, x1, y2, x2 = self.patch_coords[idx]
            img = img[y1:y2, x1:x2, :]
            mask = mask[y1:y2, x1:x2]
            mask_1 = mask_1[y1:y2, x1:x2]
            mask_3 = mask_3[y1:y2, x1:x2]
            mask_5 = mask_5[y1:y2, x1:x2]
            labels = labels[y1:y2, x1:x2]

        masks = np.stack([mask, mask_1, mask_3, mask_5, labels], axis=-1)
        if self.transform is not None:
            img, masks = self.transform(img, masks)

        labels = masks[..., -1]
        centroids, vectors, areas = helper.get_centroids_vectors_areas(labels, centroid_size=5)
        masks[..., -1] = centroids
        masks = np.moveaxis(masks, -1, 0)
        sample = {'image': img, 'masks': masks, 'vectors': vectors, 'areas': areas}
        return sample


#-----------------------------------------------------------------------
def run_check_dataset():
    ids = ['TCGA-18-5592-01Z-00-DX1']
    dataset = MODatasetDouble('../../MoNuSeg Training Data', ids, num_patches=2, patch_size=256)

    for n in range(len(dataset)):
        sample = dataset[n]
        in_cmap = colors.ListedColormap(['black', '#7CFC00'])
        in_cmap = in_cmap(np.arange(2))
        in_cmap[:, -1] = np.linspace(0, 1, 2)
        in_cmap = colors.ListedColormap(in_cmap)
        bn_cmap = colors.ListedColormap(['black', '#FF0000'])
        bn_cmap = bn_cmap(np.arange(2))
        bn_cmap[:, -1] = np.linspace(0, 1, 2)
        bn_cmap = colors.ListedColormap(bn_cmap)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.imshow(sample['image'])
        plt.imshow(sample['masks'][0], cmap=in_cmap, alpha=0.5)
        plt.imshow(sample['masks'][-1], cmap=bn_cmap, alpha=0.5)
        plt.show()
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_check_dataset()
    print('\nsuccess!')
