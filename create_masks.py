from xml.dom import minidom
from shapely.geometry import Polygon, Point
from common import *
from consts import *

def fill_inside(polygon, mask):
    for x in range(int(round(polygon.bounds[0])), min(int(round(polygon.bounds[2])), mask.shape[1] - 1)):
        for y in range(int(round(polygon.bounds[1])), min(int(round(polygon.bounds[3])), mask.shape[0] - 1)):
            if Point(x, y).intersects(polygon):
                mask[y, x] = 1

def fill_boundary(polygon, mask):
    ex, ey = polygon.exterior.xy
    for x, y in zip(ex, ey):
        mask[min(int(round(y)), mask.shape[0] - 1), min(int(round(x)), mask.shape[1] - 1)] = 1


############################# Create Masks ##################################
mask_path = os.path.join(INPUT_DIR, MASKS_DIR)
if not os.path.exists(mask_path):
    os.makedirs(mask_path)

image_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
for image_id in image_ids:
    mask_path = os.path.join(INPUT_DIR, MASKS_DIR, image_id+'.png')
    if os.path.isfile(mask_path):
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
    for region in regions:
        poly = Polygon(region)
        fill_boundary(poly, boundary_mask)
        fill_inside(poly, inside_mask)
    boundary_mask = skmorph.dilation(boundary_mask, skmorph.square(3))
    inside_mask = (np.logical_and(inside_mask, 1 - boundary_mask)).astype(int)
    mask = inside_mask * INSIDE_VALUE + boundary_mask * BOUNDARY_VALUE
    cv2.imwrite(mask_path, mask)
