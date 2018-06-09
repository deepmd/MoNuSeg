from common import *
from consts import *

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
