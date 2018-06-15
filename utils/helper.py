from common import *
from skimage import measure

def get_spaced_colors(n, cmap=None):
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


def get_border_and_centroid_vectors(labeled_mask):
    (num_rows, num_cols) = labeled_mask.shape
    vectors = np.zeros((num_rows, num_cols, 4))
    for label in range(1, np.max(labeled_mask)+1):
        temp_mask = (labeled_mask == label).astype(np.uint8)

        # create vectors to borders of regions
        inds = ndimage.morphology.distance_transform_edt(temp_mask, return_distances=False, return_indices=True)
        vectors[:, :, 0] += np.expand_dims(np.arange(0, num_rows), axis=1) - inds[0]
        vectors[:, :, 1] += np.expand_dims(np.arange(0, num_cols), axis=0) - inds[1]
        # if norm:
        #     vectors[:, :, 0] = vectors[:, :, 0] / (np.linalg.norm(vectors[:, :, 0], axis=0, keepdims=True) + 1e-5)
        #     vectors[:, :, 1] = vectors[:, :, 1] / (np.linalg.norm(vectors[:, :, 1], axis=0, keepdims=True) + 1e-5)

        # border_vectors = np.array([
        #     np.expand_dims(np.arange(0, relabeled_mask.shape[0]), axis=1) - inds[0],
        #     np.expand_dims(np.arange(0, relabeled_mask.shape[1]), axis=0) - inds[1]])

        # border_vector_norm = border_vector / (np.linalg.norm(border_vector, axis=0, keepdims=True) + 1e-5)
        # res_crop[:, :, 0] = border_vector_norm[0]
        # res_crop[:, :, 1] = border_vector_norm[1]

        # create vectors to centroids of regions
        region_props = measure.regionprops(temp_mask)
        for props in region_props:
            y, x = props.centroid
            vectors[:, :, 2] += np.multiply(np.expand_dims(x - np.arange(0, num_rows), axis=1), temp_mask)
            vectors[:, :, 3] += np.multiply(np.expand_dims(y - np.arange(0, num_cols), axis=0), temp_mask)

    return vectors