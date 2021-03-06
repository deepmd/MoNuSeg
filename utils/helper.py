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


def get_centroids_areas(labels, centroid_size=3, get_areas=True):
    (num_rows, num_cols) = labels.shape[0], labels.shape[1]
    centroids = np.zeros((num_rows, num_cols)).astype(np.uint8)
    centroid_range = math.floor((centroid_size-1) / 2)
    total_area = num_rows * num_cols
    areas = np.zeros((num_rows, num_cols), dtype=np.float) if get_areas else None

    for index in range(labels.shape[-1]):
        mask = (labels[..., index] > 0).astype(np.uint8)

        if np.count_nonzero(mask) != 0:
            # create vectors to centers of masses
            center_of_mass = ndimage.measurements.center_of_mass(mask)
            (xc, yc) = (int(round(center_of_mass[0])), int(round(center_of_mass[1])))
            centroids[xc - centroid_range:xc + centroid_range + 1, yc - centroid_range:yc + centroid_range + 1] = 1

            # calculate area of mass regions
            if get_areas:
                region_props = skimage.measure.regionprops(mask)
                for props in region_props:
                    areas += (props.area / total_area) * mask

    return centroids, areas if get_areas else centroids


def get_centroids(labels, centroid_size=3):
    return get_centroids_areas(labels, centroid_size, False)[0]


def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    # target = Variable(target)

    return target

# def get_distance_transform_based_weight_map(centroids, beta=30):
#
#     # whole_size = H*W
#     # UnblanceW = np.ones((num_classes, H, W), dtype=np.float)
#     # for i in range(num_classes):
#     #     fg_mask = np.squeeze(masks == i).astype(np.uint8)
#     #     bg_mask = 1 - fg_mask
#     #
#     #     g0_weight = whole_size / (np.count_nonzero(bg_mask) + whole_size)
#     #     g1_weight = whole_size / (np.count_nonzero(fg_mask) + whole_size)
#     #
#     #     UnblanceW[i, ...] = g0_weight * bg_mask + g1_weight * fg_mask
#
#     C, H, W = centroids.shape
#     DWM = np.ones((H, W), dtype=np.float)
#     # fg_mask = np.squeeze(masks == 2).astype(np.uint8)
#     fg_mask = np.squeeze(centroids).astype(np.uint8)
#     bg_mask = 1 - fg_mask
#     dists = ndimage.morphology.distance_transform_edt(bg_mask)
#     DWM = (1 - np.minimum(dists / beta, 1))
#
#     return DWM


def pad(img, pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]
