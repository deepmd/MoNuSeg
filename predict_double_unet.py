from common import *
from consts import *
from utils import init
from utils.metrics import aggregated_jaccard
from utils import helper
from utils.prediction import predict
from models.double_unet import DoubleUNet
from skimage import segmentation as skseg

init.set_results_reproducible()
init.init_torch()

############################# PostProcessing ##################################
def post_processing_watershed(pred):
    mask = pred[0] >= 0.5
    centroids = pred[1] >= 0.5
    mask = skmorph.remove_small_holes(mask, mask.shape[0], connectivity=mask.shape[0])
    markers = skmorph.label(centroids, connectivity=1)
    distance = ndimage.distance_transform_edt(mask)
    labels = skmorph.watershed(-distance, markers=markers, mask=mask)
    # labels = skmorph.watershed(pred[0], markers=markers, mask=mask)
    return labels

def post_processing_randomwalk(pred):
    mask = pred[0] >= 0.5
    centroids = pred[1] >= 0.5
    mask = skmorph.remove_small_holes(mask, mask.shape[0], connectivity=mask.shape[0])
    markers = skmorph.label(centroids, connectivity=1)
    labels = skseg.random_walker(mask, markers, beta=10, mode='bf')
    labels = labels * mask
    return labels

########################### Config Predict ##############################
net = DoubleUNet(DOUBLE_UNET_CONFIG).cuda()
weight_path = os.path.join(WEIGHTS_DIR, 'double-unet-0.6726.pth')
net.load_state_dict(torch.load(weight_path))

def model(img):
    _, pred = net(img)
    return pred

sum_agg_jac = 0
for test_id in TEST_IDS:
    img_path = os.path.join(INPUT_DIR, IMAGES_DIR, test_id+'.tif')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred = predict(model, img, 256, 256, 64, 64)
    pred_labels = post_processing_watershed(pred)
    num_labels = np.max(pred_labels)
    colored_labels = \
        skimage.color.label2rgb(pred_labels, colors=helper.get_spaced_colors(num_labels)).astype(np.uint8)
    pred_labels_path = os.path.join(OUTPUT_DIR, 'DUNET', LABELS_DIR, test_id)
    pred_colored_labels_path = os.path.join(OUTPUT_DIR, 'DUNET', test_id+'.png')
    np.save(pred_labels_path, pred_labels)
    bgr_labels = cv2.cvtColor(colored_labels, cv2.COLOR_RGB2BGR)
    cv2.imwrite(pred_colored_labels_path, bgr_labels)

    plt.imshow(img)
    plt.imshow(colored_labels, alpha=0.5)
    # centroids = pred[1] >= 0.5
    # markers = skmorph.label(centroids, connectivity=1)
    # plt.imshow(markers, alpha=0.5)
    plt.show()
    cv2.waitKey(0)

    labels_path = os.path.join(INPUT_DIR, LABELS_DIR, test_id+'.npy')
    gt_labels = np.load(labels_path)
    agg_jac = aggregated_jaccard(pred_labels, gt_labels)
    sum_agg_jac += agg_jac
    print('{}\'s Aggregated Jaccard Index: {:.4f}'.format(test_id, agg_jac))

print('--------------------------------------')
print('Mean Aggregated Jaccard Index: {:.4f}'.format(sum_agg_jac/len(TEST_IDS)))
