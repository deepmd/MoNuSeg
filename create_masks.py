from xml.dom import minidom
from shapely.geometry import Polygon, Point
from skimage import color, io, segmentation
from common import *
from consts import *
from utils import helper


def fill_inside(polygon, mask, label_val):
    for x in range(int(round(polygon.bounds[0])), min(int(round(polygon.bounds[2])), mask.shape[1] - 1)):
        for y in range(int(round(polygon.bounds[1])), min(int(round(polygon.bounds[3])), mask.shape[0] - 1)):
            if Point(x, y).intersects(polygon):
                mask[y, x] = label_val


def create_masks():
    paths = [os.path.join(INPUT_DIR, MASKS_DIR),
             os.path.join(INPUT_DIR, INSIDE_MASKS_DIR),
             os.path.join(INPUT_DIR, BOUNDARY_MASKS_DIR),
             os.path.join(INPUT_DIR, LABELS_DIR),
             os.path.join(INPUT_DIR, COLORED_LABELS_DIR),
             os.path.join(INPUT_DIR, BOUNDARY_COLORED_LABELS_DIR)]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    image_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
    for image_id in image_ids:
        all_exists = all([glob.glob(os.path.join(path, image_id+'*')) for path in paths])
        if all_exists:
            continue
        img_path = os.path.join(INPUT_DIR, IMAGES_DIR, image_id+'.tif')
        img = cv2.imread(img_path)
        annotation_path = os.path.join(INPUT_DIR, ANNOTATIONS_DIR, image_id+'.xml')
        xml = minidom.parse(annotation_path)
        regions_ = xml.getElementsByTagName("Region")
        regions = []
        for region in regions_:
            vertices = region.getElementsByTagName('Vertex')
            if len(vertices) < 3:
                continue #ToDo
            coords = np.zeros((len(vertices), 2))
            for i, vertex in enumerate(vertices):
                coords[i][0] = vertex.attributes['X'].value
                coords[i][1] = vertex.attributes['Y'].value
            regions.append(coords)

        # Assign pixels belong to multi region to larger region
        labeled_mask_by_area = np.zeros(img.shape[:-1])
        for i, region in enumerate(regions):
            poly = Polygon(region)
            temp_mask = np.zeros(img.shape[:-1])
            area = poly.area
            while area in np.unique(labeled_mask_by_area):
                area += .25
            fill_inside(poly, temp_mask, area)
            labeled_mask_by_area = np.maximum.reduce([labeled_mask_by_area, temp_mask])

        relabeled_mask = np.zeros_like(labeled_mask_by_area, dtype=int)
        for index, value in enumerate(np.unique(labeled_mask_by_area)):
                relabeled_mask[labeled_mask_by_area == value] = index

        # relabeled_mask, _, _ = segmentation.relabel_sequential(labeled_mask_by_area)

        mask = (relabeled_mask > 0).astype(np.uint8)

        boundary_mask = segmentation.find_boundaries(relabeled_mask, mode='inner').astype(np.uint8)

        inside_mask = np.multiply(mask, (1 - boundary_mask))

        colored_labels = color.label2rgb(relabeled_mask,
                                         colors=helper.get_spaced_colors(np.max(relabeled_mask)),
                                         bg_label=0).astype(np.uint8)

        boundary_colored_labels = segmentation.mark_boundaries(colored_labels,
                                                               relabeled_mask,
                                                               color=(1, 1, 1),
                                                               mode='inner')

        masks_path = os.path.join(INPUT_DIR, MASKS_DIR, image_id+'.png')
        labels_path = os.path.join(INPUT_DIR, LABELS_DIR, image_id+'.npy')
        inside_mask_path = os.path.join(INPUT_DIR, INSIDE_MASKS_DIR, image_id+'.png')
        boundary_mask_path = os.path.join(INPUT_DIR, BOUNDARY_MASKS_DIR, image_id+'.png')
        colored_labeled_mask_path = os.path.join(INPUT_DIR, COLORED_LABELS_DIR, image_id+'.png')
        boundary_colored_labeled_mask_path = os.path.join(INPUT_DIR, BOUNDARY_COLORED_LABELS_DIR, image_id+'.png')

        io.imsave(masks_path, mask * 255)
        np.save(labels_path, relabeled_mask)
        io.imsave(inside_mask_path, inside_mask * 255)
        io.imsave(boundary_mask_path, boundary_mask * 255)
        io.imsave(colored_labeled_mask_path, colored_labels)
        io.imsave(boundary_colored_labeled_mask_path, boundary_colored_labels)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    create_masks()
    print('\nsuccess!')
