from common import *
from utils import extract_patches


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
    pred_patches = None
    for id, sample in enumerate(patches_imgs_test):
        sample = transforms.ToTensor()(sample)
        sample = torch.unsqueeze(sample, 0)
        sample = torch.tensor(sample, dtype=torch.float).cuda()
        pred = model(sample)
        pred = F.sigmoid(pred)
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
