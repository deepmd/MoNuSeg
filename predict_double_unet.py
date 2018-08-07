from common import *
from consts import *
from utils import init
from utils.metrics import aggregated_jaccard, dice_index
from utils import helper
from utils.prediction import predict
from models.unet import DoubleUNet, DoubleWiredUNet, DoubleWiredUNet_3d, TripleUNet, TripleWiredUNet, DUNet, DWiredUNet
from models.vgg_unet import VGG_DWired_UNet16
from skimage import segmentation as skseg
import scipy.io as sio

init.set_results_reproducible()
init.init_torch()


############################# PostProcessing ##################################
def post_processing_watershed(pred, dilation=None):
    mask = pred[0] >= 0.5
    centroids = pred[-1] >= 0.5
    mask = skmorph.remove_small_holes(mask, mask.shape[0], connectivity=mask.shape[0])
    markers = skmorph.label(centroids, connectivity=1)
    distance = ndimage.distance_transform_edt(mask)
    labels = skmorph.watershed(-distance, markers=markers, mask=mask)
    # labels = skmorph.watershed(pred[0], markers=markers, mask=mask)
    if dilation is not None:
        labels = skmorph.dilation(labels, skmorph.disk(dilation))
    return labels


def post_processing_randomwalk(pred, dilation=None):
    mask = pred[0] >= 0.5
    centroids = pred[-1] >= 0.5
    mask = skmorph.remove_small_holes(mask, mask.shape[0], connectivity=mask.shape[0])
    markers = skmorph.label(centroids, connectivity=1)
    labels = skseg.random_walker(mask, markers, beta=10, mode='bf')
    labels = labels * mask
    if dilation is not None:
        labels = skmorph.dilation(labels, skmorph.disk(dilation))
    return labels


########################### Predicting ##############################
def do_prediction(net, output_path, test_ids, patch_size, stride, dilation, gate_image, masking,
                  post_processing, normalize_img, visualize=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, LABELS_DIR))

    # def model(img, mask=None):
    #     if masking:
    #         outputs = net(img, mask)
    #     else:
    #         outputs = net(img)
    #     return F.sigmoid(outputs[-1])

    def model(img):
        outputs1, outputs2 = net(img)
        pred = F.log_softmax(outputs1, dim=1)
        inside_mask = (pred.argmax(dim=1, keepdim=True) == 1).float()
        centroids = F.sigmoid(outputs2)
        return torch.cat((inside_mask, centroids), 1)

    sum_agg_jac = 0
    sum_dice = 0
    for test_id in test_ids:
        img_path = os.path.join(INPUT_DIR, IMAGES_DIR, test_id+'.tif')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if gate_image or masking:
            mask_path = os.path.join(INPUT_DIR, MASKS_DIR, test_id+'.png')
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255

        if gate_image:
            img = img * np.repeat(gt_mask[:, :, np.newaxis], img.shape[-1], axis=2)

        pred = predict(model, img, patch_size, patch_size, stride, stride, normalize_img=normalize_img) if not masking else \
               predict(model, img, patch_size, patch_size, stride, stride, normalize_img=normalize_img, mask=gt_mask)
        pred_labels = post_processing(pred, dilation=dilation)
        num_labels = np.max(pred_labels)
        colored_labels = \
            skimage.color.label2rgb(pred_labels, colors=helper.get_spaced_colors(num_labels)).astype(np.uint8)
        pred_labels_path = os.path.join(output_path, LABELS_DIR, test_id)
        pred_colored_labels_path = os.path.join(output_path, test_id+'.png')
        np.save(pred_labels_path, pred_labels)
        sio.savemat(pred_labels_path + '.mat', {'predicted_map': pred_labels}, do_compression=True)
        bgr_labels = cv2.cvtColor(colored_labels, cv2.COLOR_RGB2BGR)
        cv2.imwrite(pred_colored_labels_path, bgr_labels)

        if visualize:
            plt.imshow(img)
            plt.imshow(colored_labels, alpha=0.5)
            centroids = pred[-1] >= 0.5
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
        pred_mask = skmorph.dilation(pred[0], skmorph.disk(dilation)) if dilation is not None else pred[0]
        dice = dice_index(pred_mask, gt_mask)
        sum_dice += dice
        print('{}\'s Dice Index: {:.4f}'.format(test_id, dice))

    print('--------------------------------------')
    print('Mean Aggregated Jaccard Index: {:.4f}'.format(sum_agg_jac/len(test_ids)))
    print('Mean Dice Index: {:.4f}'.format(sum_dice/len(test_ids)))


########################### Config Predict ##############################
net = DUNet(D_UNET_CONFIG_11).cuda()

weight_path = os.path.join(WEIGHTS_DIR, 'test/dunet3_16_1e-03_1.1976.pth')
net.load_state_dict(torch.load(weight_path))
output_path = os.path.join(OUTPUT_DIR, 'DWUNET22')

do_prediction(net, output_path, TEST_IDS, patch_size=128, stride=32, dilation=1,
              gate_image=False, masking=False, post_processing=post_processing_watershed, normalize_img=True)