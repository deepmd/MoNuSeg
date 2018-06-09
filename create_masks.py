from xml.dom import minidom
from shapely.geometry import Polygon, Point
from common import *
from consts import *
from utils import helper

def fill_inside(polygon, mask, labels, label_val):
    for x in range(int(round(polygon.bounds[0])), min(int(round(polygon.bounds[2])), mask.shape[1] - 1)):
        for y in range(int(round(polygon.bounds[1])), min(int(round(polygon.bounds[3])), mask.shape[0] - 1)):
            if Point(x, y).intersects(polygon):
                mask[y, x] = 1
                labels[y, x] = label_val

def fill_boundary(polygon, mask):
    ex, ey = polygon.exterior.xy
    for x, y in zip(ex, ey):
        mask[min(int(round(y)), mask.shape[0] - 1), min(int(round(x)), mask.shape[1] - 1)] = 1


############################# Create Masks ##################################
masks_path = os.path.join(INPUT_DIR, MASKS_DIR)
if not os.path.exists(masks_path):
    os.makedirs(masks_path)
labels_path = os.path.join(INPUT_DIR, LABELS_DIR)
if not os.path.exists(labels_path):
    os.makedirs(labels_path)

image_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
for image_id in image_ids:
    masks_path = os.path.join(INPUT_DIR, MASKS_DIR, image_id+'.png')
    labels_path = os.path.join(INPUT_DIR, LABELS_DIR, image_id+'.png')
    if os.path.isfile(masks_path) and os.path.isfile(labels_path):
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
    inside_mask = np.zeros(img.shape[:-1])
    boundary_mask = np.zeros(img.shape[:-1])
    labels = np.zeros(img.shape[:-1])
    for i, region in enumerate(regions):
        poly = Polygon(region)
        fill_boundary(poly, boundary_mask)
        fill_inside(poly, inside_mask, labels, i+1)
    boundary_mask = skmorph.dilation(boundary_mask, skmorph.square(3))
    inside_mask = (np.logical_and(inside_mask, 1 - boundary_mask)).astype(int)
    mask = inside_mask * INSIDE_VALUE + boundary_mask * BOUNDARY_VALUE
    cv2.imwrite(masks_path, mask)
    num_labels = np.max(len(regions))
    colored_labels = \
        skimage.color.label2rgb(labels, colors=helper.get_spaced_colors(num_labels)).astype(np.uint8)
    rgb_labels = cv2.cvtColor(colored_labels, cv2.COLOR_RGB2BGR)
    cv2.imwrite(labels_path, rgb_labels)
