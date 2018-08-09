from common import *
from consts import *
from utils import extract_patches


def predict(model, test_img, patch_height, patch_width, stride_height, stride_width, average_mode=True,
            normalize_img=False, mask=None, image_scales_number=1):
    test_img = test_img[np.newaxis, ...]
    test_img = np.moveaxis(test_img, 3, 1)
    full_img_height = test_img.shape[2]
    full_img_width = test_img.shape[3]
    if mask is not None:
        mask = mask[np.newaxis, np.newaxis, ...]

    if average_mode:
        patches_imgs_test, new_height, new_width = extract_patches.get_data_testing_overlap(
            test_imgs=test_img,
            patch_height=patch_height, patch_width=patch_width,
            stride_height=stride_height, stride_width=stride_width)
    else:
        patches_imgs_test, N_patches_h, N_patches_w = extract_patches.get_data_testing(
            test_imgs=test_img, patch_height=patch_height, patch_width=patch_width)

    if mask is None:
        patches_masks = [None]*patches_imgs_test.shape[0]
    elif average_mode:
        patches_masks, _, _ = extract_patches.get_data_testing_overlap(
            test_imgs=mask,
            patch_height=patch_height, patch_width=patch_width,
            stride_height=stride_height, stride_width=stride_width)
    else:
        patches_masks, _, _ = extract_patches.get_data_testing(
            test_imgs=mask, patch_height=patch_height, patch_width=patch_width)

    # Calculate the predictions
    patches_imgs_test = np.moveaxis(patches_imgs_test, 1, 3)
    out_shape = patches_imgs_test.shape
    pred_patches = None
    for id, (sample, mask) in enumerate(zip(patches_imgs_test, patches_masks)):
        samples = []
        for i in range(0, image_scales_number):
            resized_sample = sample
            if i > 0:
                scale = 2**i
                resized_sample = cv2.resize(sample, (sample.shape[0]//scale, sample.shape[1]//scale),
                                            interpolation=cv2.INTER_LINEAR)
            resized_sample = transforms.ToTensor()(resized_sample.astype(np.uint8))
            if normalize_img:
                resized_sample = transforms.Normalize(IMAGES_MEAN, IMAGES_STD)(resized_sample)
            resized_sample = torch.unsqueeze(resized_sample, 0)
            resized_sample = torch.tensor(resized_sample, dtype=torch.float).cuda()
            samples.append(resized_sample)
        samples = samples[0] if len(samples) == 1 else samples

        if mask is None:
            pred = model(samples)
        else:
            mask = torch.tensor(mask, dtype=torch.float).cuda()
            pred = model(samples, mask)
        if pred_patches is None:
            pred_patches = np.zeros((out_shape[0], pred.shape[1], out_shape[1], out_shape[2]), dtype=np.float32)
        pred_patches[id, ...] = pred.data.cpu().numpy()

    # Recompose patches into image
    if average_mode:
        pred_imgs = extract_patches.recompose_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    else:
        pred_imgs = extract_patches.recompose(pred_patches, N_patches_h, N_patches_w)  # predictions

    # back to original dimensions
    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]

    return pred_imgs[0]
