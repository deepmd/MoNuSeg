from common import *
from skimage import io


def get_spaced_colors(n, cmap=None):
    n = 1 if n == 0 else n
    if cmap is None:
        max_value = 255**3
        interval = int(max_value/n)
        colors = [hex(I)[2:].zfill(6) for I in range(interval, max_value, interval)]
        colors = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
    else:
        cmap = matplotlib.cm.get_cmap(cmap)
        colors = [(cmap(x)[0]*255, cmap(x)[1]*255, cmap(x)[2]*255) for x in np.arange(0, 1, 1/n)]
    random.shuffle(colors)
    colors = [(0, 0, 0)] + colors
    return colors


def rgb2label(img):
    colors_coords = {}
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            color = tuple(img[y, x, :])
            if color == (0, 0, 0):
                continue
            if color not in colors_coords.keys():
                colors_coords[color] = []
            colors_coords[color] += [(y, x)]
    labels = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
    for i, color in enumerate(colors_coords.keys()):
        for y, x in colors_coords[color]:
            labels[y, x] = i+1
    return labels


def get_centroids_vectors_areas(labels, centroid_size=3, vectors_3d=False):
    (num_rows, num_cols) = labels.shape[0], labels.shape[1]
    areas = np.zeros((num_rows, num_cols), dtype=np.float)
    centroids = np.zeros((num_rows, num_cols)).astype(np.uint8)
    total_area = num_rows * num_cols
    vectors = np.zeros((4, num_rows, num_cols)) if not vectors_3d else np.zeros((6, num_rows, num_cols))
    centroid_range = math.floor((centroid_size-1) / 2)

    for index in range(labels.shape[-1]):
        mask = (labels[..., index] > 0).astype(np.uint8)

        if np.count_nonzero(mask) != 0:
            # create vectors to borders of regions
            inds = ndimage.morphology.distance_transform_edt(mask, return_distances=False, return_indices=True)
            (x_nonzeros, y_nonzeros) = np.nonzero(mask)
            xs = inds[0, x_nonzeros, y_nonzeros]
            ys = inds[1, x_nonzeros, y_nonzeros]
            vectors[0, x_nonzeros, y_nonzeros] = x_nonzeros - xs
            vectors[1, x_nonzeros, y_nonzeros] = y_nonzeros - ys

            # create vectors to centers of masses
            center_of_mass = ndimage.measurements.center_of_mass(mask)
            (xc, yc) = (int(round(center_of_mass[0])), int(round(center_of_mass[1])))
            centroids[xc - centroid_range:xc + centroid_range + 1, yc - centroid_range:yc + centroid_range + 1] = 1
            vectors[2, x_nonzeros, y_nonzeros] = x_nonzeros - xc
            vectors[3, x_nonzeros, y_nonzeros] = y_nonzeros - yc

            # calculate area of mass regions
            region_props = skimage.measure.regionprops(mask)
            for props in region_props:
                areas += (props.area / total_area) * mask

    norms = np.concatenate(
        [np.tile(np.linalg.norm(vectors[:2], 2, 0), (2, 1, 1)),
         np.tile(np.linalg.norm(vectors[2:], 2, 0), (2, 1, 1))])
    norms[norms == 0] = 1  # preventing division by zero
    vectors[:4] = vectors[:4] / norms

    if vectors_3d:
        vectors[4] = vectors[3]
        vectors[3] = vectors[2]
        vectors[2] = 0
        background = (areas == 0).astype(np.uint8)
        (x_nonzeros, y_nonzeros) = np.nonzero(background)
        vectors[2, x_nonzeros, y_nonzeros] = vectors[5, x_nonzeros, y_nonzeros] = 1

    return centroids, vectors, areas