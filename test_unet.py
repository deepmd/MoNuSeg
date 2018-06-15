from common import *
from consts import *
from utils import init
from utils import extract_patches
from utils import helper
from utils.metrics import aggregated_jaccard
from models.unet import UNet

init.set_results_reproducible()
init.init_torch()

############################# Predicting ##################################
def predict(model, test_img, patch_height, patch_width, stride_height, stride_width, average_mode=True):
    test_img = test_img[np.newaxis, ...]
    test_img = np.moveaxis(test_img, 3, 1)
    full_img_height = test_img.shape[2]
    full_img_width = test_img.shape[3]

    if average_mode:
        patches_imgs_test, new_height, new_width = extract_patches.get_data_testing_overlap(
            test_imgs=test_img,
            patch_height=patch_height,
            patch_width=patch_width,
            stride_height=stride_height,
            stride_width=stride_width
        )
    else:
        patches_imgs_test, N_patches_h, N_patches_w = extract_patches.get_data_testing(
            test_imgs=test_img,
            patch_height=patch_height,
            patch_width=patch_width,
        )

    # Calculate the predictions
    patches_imgs_test = np.moveaxis(patches_imgs_test, 1, 3)
    out_shape = patches_imgs_test.shape
    pred_patches = np.zeros((out_shape[0], 3, out_shape[1], out_shape[2]), dtype=np.float32)
    for id, sample in enumerate(patches_imgs_test):
        sample = transforms.ToTensor()(sample)
        sample = torch.unsqueeze(sample, 0)
        sample = torch.tensor(sample, dtype=torch.float).cuda()
        pred = model(sample)
        pred = F.sigmoid(pred)
        pred_patches[id, ...] = pred.data.cpu().numpy()

    # ========== Elaborate and visualize the predicted images ====================
    if average_mode:
        pred_imgs = extract_patches.recompose_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    else:
        pred_imgs = extract_patches.recompose(pred_patches, N_patches_h, N_patches_w)  # predictions

    # back to original dimensions
    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]

    return pred_imgs[0]

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
weight_path = os.path.join(WEIGHTS_DIR, 'unet-0.7819.pth')
net.load_state_dict(torch.load(weight_path))
sum_agg_jac = 0

for test_id in TEST_IDS:
    img_path = os.path.join(INPUT_DIR, IMAGES_DIR, test_id+'.tif')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred = predict(net, img, 256, 256, 64, 64)
    pred_labels = post_processing(pred)
    num_labels = np.max(pred_labels)
    colored_labels = \
        skimage.color.label2rgb(pred_labels, colors=helper.get_spaced_colors(num_labels)).astype(np.uint8)
    pred_labels_path = os.path.join(OUTPUT_DIR, 'UNET', LABELS_DIR, test_id)
    pred_colored_labels_path = os.path.join(OUTPUT_DIR, 'UNET', COLORED_LABELS_DIR, test_id+'.png')
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

print('--------------------------------------')
print('Mean Aggregated Jaccard Index: {:.4f}'.format(sum_agg_jac/len(TEST_IDS)))
