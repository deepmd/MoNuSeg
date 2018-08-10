from common import *
from consts import *
from utils import init
from utils.metrics import aggregated_jaccard, dice_index
from utils import helper
from utils.prediction import predict
from models.vgg_unet import VGG_UNet16, VGG_Holistic_UNet16
from scipy.ndimage import morphology
import scipy.io as sio

init.set_results_reproducible()
init.init_torch()

############################# PostProcessing ##################################
def post_processing_watershed(pred, dilation=None):
    mask = np.argmax(pred, axis=0) >= 2
    centroids = np.argmax(pred, axis=0) == 3
    mask = skmorph.remove_small_holes(mask, mask.shape[0], connectivity=mask.shape[0])
    markers = skmorph.label(centroids, connectivity=1)
    distance = ndimage.distance_transform_edt(mask)
    labels = skmorph.watershed(-distance, markers=markers, mask=mask)
    if dilation is not None:
        labels = skmorph.dilation(labels, skmorph.disk(dilation))
    return labels

# def post_processing_mask(pred):
#     mask = np.argmax(pred, axis=0) == 2
#     inside_mask = morphology.binary_fill_holes(mask).astype(np.int)
#     labels = skmorph.binary_dilation(inside_mask, selem=skmorph.square(3)).astype(np.int)
#     return labels


########################### Predicting ##############################
def do_prediction(net, output_path, test_ids, patch_size, stride, post_processing, dilation, visualize=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, LABELS_DIR))

    def model(img):
        # _,_,_,_,_,outputs = net(img)
        outputs = net(img)
        # outputs = torch.cat(outputs, dim=0)
        # outputs = torch.mean(outputs, dim=0, keepdim=True)
        pred = F.softmax(outputs, dim=1)
        return pred

    sum_agg_jac = 0
    sum_dice = 0
    for test_id in test_ids:
        img_path = os.path.join(INPUT_DIR, IMAGES_DIR, test_id+'.tif')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = predict(model, img, patch_size, patch_size, stride, stride, normalize_img=True)
        pred_labels = post_processing(pred, dilation=dilation)
        num_labels = np.max(pred_labels)
        colored_labels = \
            skimage.color.label2rgb(pred_labels, colors=helper.get_spaced_colors(num_labels)).astype(np.uint8)
        pred_labels_path = os.path.join(output_path, LABELS_DIR, test_id)
        pred_colored_labels_path = os.path.join(output_path, test_id + '.png')
        np.save(pred_labels_path, pred_labels)
        sio.savemat(pred_labels_path + '.mat', {'predicted_map': pred_labels}, do_compression=True)
        bgr_labels = cv2.cvtColor(colored_labels, cv2.COLOR_RGB2BGR)
        cv2.imwrite(pred_colored_labels_path, bgr_labels)

        if visualize:
            plt.imshow(img)
            plt.imshow(colored_labels, alpha=0.5)
            centroids = np.argmax(pred, axis=0) == 3
            plt.imshow(centroids, alpha=0.5)
            plt.show()
            cv2.waitKey(0)

        labels_path = os.path.join(INPUT_DIR, LABELS_DIR, test_id+'.npy')
        gt_labels = np.load(labels_path).astype(np.int)
        agg_jac = aggregated_jaccard(pred_labels, gt_labels)
        sum_agg_jac += agg_jac
        print('{}\'s Aggregated Jaccard Index: {:.4f}'.format(test_id, agg_jac))

        mask_path = os.path.join(INPUT_DIR, MASKS_DIR, test_id+'.png')
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
        pred_mask = skmorph.dilation(np.argmax(pred, axis=0) >= 2, skmorph.disk(dilation)) if dilation is not None else \
                    np.argmax(pred, axis=0) >= 2
        dice = dice_index(pred_mask, gt_mask)
        sum_dice += dice
        print('{}\'s Dice Index: {:.4f}'.format(test_id, dice))

    print('--------------------------------------')
    print('Mean Aggregated Jaccard Index: {:.4f}'.format(sum_agg_jac/len(test_ids)))
    print('Mean Dice Index: {:.4f}'.format(sum_dice/len(test_ids)))


########################### Config Predict ##############################
net = VGG_UNet16(num_classes=4, pretrained=False).cuda()
net.eval()

weight_path = os.path.join(WEIGHTS_DIR, 'unet-0.6522.pth')
net.load_state_dict(torch.load(weight_path))
output_path = os.path.join(OUTPUT_DIR, 'VGG16')

# all_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
do_prediction(net, output_path, TEST_IDS, patch_size=128, stride=32,
              post_processing=post_processing_watershed, dilation=1)
