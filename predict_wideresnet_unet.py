from common import *
from consts import *
from utils import init
from utils.metrics import aggregated_jaccard, dice_index
from utils import helper
from utils.prediction import predict
from models.vgg_unet import VGG_UNet16, VGG_Holistic_UNet16
from scipy.ndimage import morphology
import scipy.io as sio
from models.wideresnet_unet.ternausnet2 import TernausNetV2

init.set_results_reproducible()
init.init_torch()

############################# PostProcessing ##################################
def post_processing_watershed(prediction, dilation=None, component_size=20):
    mask = (prediction[1] > MASK_THRESHOLD)
    # mask = skmorph.remove_small_holes(mask, mask.shape[0], connectivity=mask.shape[0]).astype(np.uint8)
    touching_borders = (prediction[0])

    seed = ((mask * (1 - touching_borders)) > MASK_THRESHOLD).astype(np.uint8)

    markers = skmorph.label(seed, connectivity=1)
    labels = skmorph.watershed(-mask, markers=markers, mask=mask)
    unique, counts = np.unique(labels, return_counts=True)

    for (k, v) in dict(zip(unique, counts)).items():
        if v < component_size:
            labels[labels == k] = 0

    if dilation is not None:
        # labels = skmorph.dilation(labels, skmorph.disk(dilation))
        labels = skmorph.dilation(labels, skmorph.square(dilation))

    return labels, mask

########################### Predicting ##############################
def do_prediction(net, output_path, test_ids, post_processing, dilation, visualize=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, LABELS_DIR))

    def model(img):
        # _,_,_,_,_,outputs = net(img)
        outputs = net(img)
        # outputs = torch.cat(outputs, dim=0)
        # outputs = torch.mean(outputs, dim=0, keepdim=True)
        pred = F.sigmoid(outputs)
        return pred

    sum_agg_jac = 0
    sum_dice = 0
    for test_id in test_ids:
        img_path = os.path.join(INPUT_DIR, IMAGES_DIR, test_id+'.tif')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Network contains 5 maxpool layers => input should be divisible by 2**5 = 32
        img, pads = helper.pad(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(IMAGES_MEAN, IMAGES_STD)(img)
        img = torch.unsqueeze(img, 0)
        img = torch.tensor(img, dtype=torch.float).cuda()

        prediction = np.squeeze(model(img).data.cpu().numpy())

        # left mask, right touching areas
        # plt.imshow(np.hstack([prediction[0], prediction[1]]))

        pred_labels, pred_mask = post_processing(prediction, dilation=dilation, component_size=0)
        pred_labels = helper.unpad(pred_labels, pads)
        pred_mask = helper.unpad(pred_mask, pads)


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
            plt.show()
            cv2.waitKey(0)

        labels_path = os.path.join(INPUT_DIR, LABELS_DIR, test_id+'.npy')
        gt_labels = np.load(labels_path).astype(np.int)
        agg_jac = aggregated_jaccard(pred_labels, gt_labels)
        sum_agg_jac += agg_jac
        print('{}\'s Aggregated Jaccard Index: {:.4f}'.format(test_id, agg_jac))

        mask_path = os.path.join(INPUT_DIR, MASKS_DIR, test_id+'.png')
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
        pred_mask = skmorph.binary_dilation(pred_mask, skmorph.square(dilation)) if dilation is not None else prediction
        dice = dice_index(pred_mask, gt_mask)
        sum_dice += dice
        print('{}\'s Dice Index: {:.4f}'.format(test_id, dice))

    print('--------------------------------------')
    print('Mean Aggregated Jaccard Index: {:.4f}'.format(sum_agg_jac/len(test_ids)))
    print('Mean Dice Index: {:.4f}'.format(sum_dice/len(test_ids)))


########################### Config Predict ##############################
net = TernausNetV2(num_classes=2, num_input_channels=3).cuda()
net.eval()

weight_path = os.path.join(WEIGHTS_DIR, 'final2/unet_29_0.7366.pth')
net.load_state_dict(torch.load(weight_path))
output_path = os.path.join(OUTPUT_DIR, 'WIDERES1')

# all_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
do_prediction(net, output_path, TEST_IDS, post_processing=post_processing_watershed, dilation=3, visualize=False)
