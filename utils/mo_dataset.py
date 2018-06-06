from common import *
from consts import *

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
            else:
                img_path = os.path.join(self.root_dir, IMAGES_DIR, self.ids[0] + '.tif')
                img = cv2.imread(img_path)
                self.patch_coords = np.zeros((len(self.ids), 4), dtype=np.int)
                self.patch_coords[:, 0] = np.random.randint(0, img.shape[:-1][0]-patch_size-1, (len(self.ids)))
                self.patch_coords[:, 1] = np.random.randint(0, img.shape[:-1][1]-patch_size-1, (len(self.ids)))
                self.patch_coords[:, 2] = self.patch_coords[:, 0] + patch_size
                self.patch_coords[:, 3] = self.patch_coords[:, 1] + patch_size
                np.savetxt(patch_info_path, self.patch_coords, delimiter=',', fmt='%d')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, IMAGES_DIR, self.ids[idx]+'.tif')
        img = cv2.imread(img_path)

        mask_path = os.path.join(self.root_dir, MASKS_DIR, self.ids[idx]+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        inside_mask = (mask == INSIDE_VALUE).astype(int)
        boundary_mask = (mask == BOUNDARY_VALUE).astype(int)
        background_mask = 1 - np.logical_or(inside_mask, boundary_mask)

        if self.patch_coords is not None:
            y1, x1, y2, x2 = self.patch_coords[idx]
            img             = img[y1:y2, x1:x2, :]
            inside_mask     = inside_mask[y1:y2, x1:x2]
            boundary_mask   = boundary_mask[y1:y2, x1:x2]
            background_mask = background_mask[y1:y2, x1:x2]

        sample = {'image': img, 'masks': (inside_mask, boundary_mask, background_mask)}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


#-----------------------------------------------------------------------
def run_check_dataset():
    ids = ['TCGA-18-5592-01Z-00-DX1']
    dataset = MODataset('../../MoNuSeg Training Data', ids, num_patches=2, patch_size=256)

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
        img = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.imshow(sample['masks'][0], cmap=in_cmap, alpha=0.5)
        plt.imshow(sample['masks'][1], cmap=bn_cmap, alpha=0.5)
        plt.show()
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_check_dataset()
    print('\nsuccess!')