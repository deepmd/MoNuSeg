from common import *


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


def get_centroids_vectors_areas(labels, centroid_size=3):
    (num_rows, num_cols) = labels.shape[0], labels.shape[1]
    centroids = np.zeros((num_rows, num_cols)).astype(np.uint8)
    areas = np.zeros((num_rows, num_cols), dtype=np.float)

    total_area = num_rows * num_cols
    vectors = np.zeros((4, num_rows, num_cols))
    centroid_range = math.floor((centroid_size-1) / 2)

    for index in range(labels.shape[-1]):
        mask = (labels[..., index] > 0).astype(np.uint8)

        # create vectors to borders of regions
        inds = ndimage.morphology.distance_transform_edt(mask, return_distances=False, return_indices=True)
        vectors[0, :, :] += np.expand_dims(np.arange(0, num_rows), axis=1) - inds[0]
        vectors[1, :, :] += np.expand_dims(np.arange(0, num_cols), axis=0) - inds[1]

        # create vectors to centers of masses
        if np.count_nonzero(mask) != 0:
            center_of_mass = ndimage.measurements.center_of_mass(mask)
            (y, x) = (int(round(center_of_mass[0])), int(round(center_of_mass[1])))
            centroids[y - centroid_range:y + centroid_range + 1, x - centroid_range:x + centroid_range + 1] = 1
            vectors[2, :, :] += np.multiply(np.expand_dims(x - np.arange(0, num_rows), axis=1), mask)
            vectors[3, :, :] += np.multiply(np.expand_dims(y - np.arange(0, num_cols), axis=0), mask)

            region_props = skimage.measure.regionprops(mask)
            for props in region_props:
                areas += (props.area / total_area) * mask

    norms = np.concatenate(
        [np.tile(np.linalg.norm(vectors[:2], 2, 0), (2, 1, 1)),
         np.tile(np.linalg.norm(vectors[2:], 2, 0), (2, 1, 1))])
    norms[norms == 0] = 1  # preventing division by zero
    vectors = vectors / norms

    return centroids, vectors, areas


# def get_centroids_vectors_areas(labeled_mask, centroid_size=3):
#     centroids = np.zeros_like(labeled_mask)
#     areas = np.zeros_like(labeled_mask, dtype=np.float)
#     (num_rows, num_cols) = labeled_mask.shape
#     total_area = num_rows * num_cols
#     vectors = np.zeros((4, num_rows, num_cols))
#     centroid_range = math.floor((centroid_size-1) / 2)
#
#     for label in range(1, len(np.unique(labeled_mask))):
#         temp_mask = (labeled_mask == label).astype(np.uint8)
#
#         # create vectors to borders of regions
#         inds = ndimage.morphology.distance_transform_edt(temp_mask, return_distances=False, return_indices=True)
#         vectors[0, :, :] += np.expand_dims(np.arange(0, num_rows), axis=1) - inds[0]
#         vectors[1, :, :] += np.expand_dims(np.arange(0, num_cols), axis=0) - inds[1]
#
#         # create vectors to centers of masses
#         center_of_mass = ndimage.measurements.center_of_mass(temp_mask)
#         (y, x) = (int(round(center_of_mass[0])), int(round(center_of_mass[1])))
#         centroids[y - centroid_range:y + centroid_range + 1, x - centroid_range:x + centroid_range + 1] = 1
#         vectors[2, :, :] += np.multiply(np.expand_dims(x - np.arange(0, num_rows), axis=1), temp_mask)
#         vectors[3, :, :] += np.multiply(np.expand_dims(y - np.arange(0, num_cols), axis=0), temp_mask)
#
#         region_props = skimage.measure.regionprops(temp_mask)
#         for props in region_props:
#             areas += (props.area / total_area) * temp_mask
#
#     norms = np.concatenate(
#         [np.tile(np.linalg.norm(vectors[:2], 2, 0), (2, 1, 1)),
#          np.tile(np.linalg.norm(vectors[2:], 2, 0), (2, 1, 1))])
#     norms[norms == 0] = 1  # preventing division by zero
#     vectors = vectors / norms
#
#     return centroids, vectors, areas