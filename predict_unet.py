from common import *
from consts import *
from utils import init
from utils.prediction import predict
from utils.metrics import dice_index
from models.unet.vgg_unet_model import UNet16
from models.unet import UNet
from skimage import io
from scipy.ndimage import morphology

init.set_results_reproducible()
init.init_torch()

############################# Predicting ##################################
def post_processing(pred):
    # inside_pred = pred[1] >= 0.5
    # boundary_pred = pred[2] >= 0.5
    # inside_pred = np.logical_and(inside_pred, np.logical_not(boundary_pred))
    # inside_pred = skmorph.remove_small_holes(inside_pred, inside_pred.shape[0], connectivity=inside_pred.shape[0])
    # labels = skmorph.label(inside_pred, connectivity=1)
    # labels = skmorph.dilation(labels)
    inside_pred = morphology.binary_fill_holes(np.squeeze(pred)).astype(np.int)
    labels = skmorph.binary_dilation(inside_pred, selem=skmorph.square(3)).astype(np.int)
    return labels


########################### Config Predict ##############################
# net = UNet(UNET_CONFIG).cuda()
# net = Res_UNet(layers=34, out_channels=3).cuda()
net = UNet16(num_classes=3, pretrained=False).cuda()
# net = LinkNet34(num_classes=3, pretrained=True).cuda()

weight_path = os.path.join(WEIGHTS_DIR, 'UNET3/unet-0.6693.pth')
net.load_state_dict(torch.load(weight_path))
output_path = 'UNET_VGG16_512_Mask_Less_Aug'
output_path = os.path.join(OUTPUT_DIR, output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, LABELS_DIR))

def model(img):
    _, pred = net(img)
    return pred

sum_agg_jac = 0
sum_dice = 0
# all_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
for test_id in TEST_IDS:
    img_path = os.path.join(INPUT_DIR, IMAGES_DIR, test_id+'.tif')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred = predict(net, img, 512, 512, 64, 64)
    pred_labels = post_processing(pred)
    io.imsave(os.path.join(output_path, test_id + '.png'), pred_labels*255)
    # num_labels = np.max(pred_labels)
    # colored_labels = \
    #     skimage.color.label2rgb(pred_labels, colors=helper.get_spaced_colors(num_labels)).astype(np.uint8)
    # pred_labels_path = os.path.join(output_path, LABELS_DIR, test_id)
    # pred_colored_labels_path = os.path.join(output_path, test_id+'.png')
    # np.save(pred_labels_path, pred_labels)
    # bgr_labels = cv2.cvtColor(colored_labels, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(pred_colored_labels_path, bgr_labels)

    # plt.imshow(img)
    # plt.imshow(colored_labels, alpha=0.5)
    # plt.show()
    # cv2.waitKey(0)

    # colored_labels_path = os.path.join(INPUT_DIR, COLORED_LABELS_DIR, test_id+'.png')
    # labels_img = cv2.imread(colored_labels_path)
    # gt_labels = helper.rgb2label(labels_img)
    # labels_path = os.path.join(INPUT_DIR, LABELS_DIR, test_id+'.npy')
    # gt_labels = np.load(labels_path)
    # agg_jac = aggregated_jaccard(pred_labels, gt_labels)
    # sum_agg_jac += agg_jac
    # print('{}\'s Aggregated Jaccard Index: {:.4f}'.format(test_id, agg_jac))

    mask_path = os.path.join(INPUT_DIR, INSIDE_MASKS_DIR, test_id+'.png')
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
    dice = dice_index(pred_labels, gt_mask)
    sum_dice += dice
    print('{}\'s Dice Index: {:.4f}'.format(test_id, dice))

print('--------------------------------------')
# print('Mean Aggregated Jaccard Index: {:.4f}'.format(sum_agg_jac/len(TEST_IDS)))
print('Mean Dice Index: {:.4f}'.format(sum_dice/len(TEST_IDS)))
