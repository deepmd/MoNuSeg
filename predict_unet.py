from common import *
from consts import *
from utils import init
from utils import helper
from utils.prediction import predict
from utils.metrics import aggregated_jaccard, dice_index
from models.unet import UNet

init.set_results_reproducible()
init.init_torch()

############################# Predicting ##################################
def post_processing(pred):
    inside_pred = pred[0] >= 0.5
    # boundary_pred = pred[1] >= 0.5
    # inside_pred = np.logical_and(inside_pred, np.logical_not(boundary_pred))
    inside_pred = skmorph.remove_small_holes(inside_pred, inside_pred.shape[0], connectivity=inside_pred.shape[0])
    labels = skmorph.label(inside_pred, connectivity=1)
    labels = skmorph.dilation(labels)
    return labels

########################### Config Predict ##############################
net = UNet(UNET_CONFIG).cuda()
weight_path = os.path.join(WEIGHTS_DIR, 'unet-0.4974.pth')
net.load_state_dict(torch.load(weight_path))
output_path = 'UNET'
output_path = os.path.join(OUTPUT_DIR, output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, LABELS_DIR))

def model(img):
    _, pred = net(img)
    return pred

sum_agg_jac = 0
sum_dice = 0
for test_id in TEST_IDS:
    img_path = os.path.join(INPUT_DIR, IMAGES_DIR, test_id+'.tif')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred = predict(net, img, 256, 256, 64, 64)
    pred_labels = post_processing(pred)
    num_labels = np.max(pred_labels)
    colored_labels = \
        skimage.color.label2rgb(pred_labels, colors=helper.get_spaced_colors(num_labels)).astype(np.uint8)
    pred_labels_path = os.path.join(output_path, LABELS_DIR, test_id)
    pred_colored_labels_path = os.path.join(output_path, test_id+'.png')
    np.save(pred_labels_path, pred_labels)
    bgr_labels = cv2.cvtColor(colored_labels, cv2.COLOR_RGB2BGR)
    cv2.imwrite(pred_colored_labels_path, bgr_labels)

    # plt.imshow(img)
    # plt.imshow(colored_labels, alpha=0.5)
    # plt.show()
    # cv2.waitKey(0)

    # colored_labels_path = os.path.join(INPUT_DIR, COLORED_LABELS_DIR, test_id+'.png')
    # labels_img = cv2.imread(colored_labels_path)
    # gt_labels = helper.rgb2label(labels_img)
    labels_path = os.path.join(INPUT_DIR, LABELS_DIR, test_id+'.npy')
    gt_labels = np.load(labels_path)
    agg_jac = aggregated_jaccard(pred_labels, gt_labels)
    sum_agg_jac += agg_jac
    print('{}\'s Aggregated Jaccard Index: {:.4f}'.format(test_id, agg_jac))

    mask_path = os.path.join(INPUT_DIR, MASKS_DIR, test_id+'.png')
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
    dice = dice_index(pred[0], gt_mask)
    sum_dice += dice
    print('{}\'s Dice Index: {:.4f}'.format(test_id, dice))

print('--------------------------------------')
print('Mean Aggregated Jaccard Index: {:.4f}'.format(sum_agg_jac/len(TEST_IDS)))
print('Mean Dice Index: {:.4f}'.format(sum_dice/len(TEST_IDS)))
