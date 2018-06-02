from xml.dom import minidom
from shapely.geometry import Polygon, Point
from common import *
from consts import *

INSIDE_VALUE = 1
BOUNDARY_VALUE = 0.5

class MODataset(Dataset):
    """Multi Organ Dataset"""

    def __init__(self, root_dir, ids, transform=None):
        self.root_dir = root_dir
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, IMAGES_DIR, self.ids[idx]+'.tif')
        img = cv2.imread(img_path)

        mask_path = os.path.join(self.root_dir, MASKS_DIR)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        mask_path = os.path.join(mask_path, self.ids[idx]+'.png')
        if os.path.isfile(mask_path):
            mask = cv2.imread(mask_path) / 255
            inside_mask = mask == INSIDE_VALUE
            boundary_mask = mask == BOUNDARY_VALUE
            background_mask = 1 - np.logical_or(inside_mask, boundary_mask)
        else:
            annotation_path = os.path.join(self.root_dir, ANNOTATIONS_DIR, self.ids[idx]+'.xml')
            xml = minidom.parse(annotation_path)
            regions_ = xml.getElementsByTagName("Region")
            regions = []
            for region in regions_:
                vertices = region.getElementsByTagName('Vertex')
                coords = np.zeros((len(vertices), 2))
                for i, vertex in enumerate(vertices):
                    coords[i][0] = vertex.attributes['X'].value
                    coords[i][1] = vertex.attributes['Y'].value
                regions.append(coords)
            inside_mask = np.zeros(img.shape[:-1])
            boundary_mask = np.zeros(img.shape[:-1])
            for region in regions:
                poly = Polygon(region)
                MODataset.fill_boundary(poly, boundary_mask)
                MODataset.fill_inside(poly, inside_mask)
            boundary_mask = dilation(boundary_mask, square(3))
            inside_mask = np.logical_and(inside_mask, 1 - boundary_mask)
            background_mask = 1 - np.logical_or(inside_mask, boundary_mask)
            mask = inside_mask * INSIDE_VALUE + boundary_mask * BOUNDARY_VALUE
            cv2.imwrite(mask_path, mask*255)

        sample = {'image': img, 'masks': (inside_mask, boundary_mask, background_mask)}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def fill_inside(polygon, mask):
        for x in range(int(round(polygon.bounds[0])), max(int(round(polygon.bounds[2])), mask.shape[1]-1)):
            for y in range(int(round(polygon.bounds[1])), max(int(round(polygon.bounds[3])), mask.shape[0]-1)):
                if Point(x, y).intersects(polygon):
                    mask[y, x] = INSIDE_VALUE

    @staticmethod
    def fill_boundary(polygon, mask):
        ex, ey = polygon.exterior.xy;
        for x, y in zip(ex, ey):
            mask[max(int(round(y)), mask.shape[0]-1), max(int(round(x)), mask.shape[1]-1)] = BOUNDARY_VALUE


#-----------------------------------------------------------------------
def run_check_dataset():
    ids = ['TCGA-18-5592-01Z-00-DX1']
    dataset = MODataset('../../MoNuSeg Training Data', ids)

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