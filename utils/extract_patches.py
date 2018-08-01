from common import *


# Load the original data and return the extracted patches for training/testing
def get_data_testing(test_imgs, patch_height, patch_width):
    # test
    test_imgs = paint_border(test_imgs, patch_height, patch_width)
    # print("\ntest images/masks shape:", test_imgs.shape)
    # print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))
    # print("test masks are within 0-1\n")

    # extract the TEST patches from the full images
    patches_imgs_test, N_patches_h, N_patches_w = extract_ordered(test_imgs, patch_height, patch_width)
    # print("\ntest PATCHES images/masks shape:", patches_imgs_test.shape)
    # print("test PATCHES images range (min-max): " +
    #       str(np.min(patches_imgs_test)) + ' - ' +
    #       str(np.max(patches_imgs_test)))

    return patches_imgs_test, N_patches_h, N_patches_w


# Load the original data and return the extracted patches for testing
# return the ground truth in its original shape
def get_data_testing_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width):
    # test
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    # print("\ntest images shape:", test_imgs.shape)
    # print("test images range (min-max): " + str(np.min(test_imgs)) + '-' + str(np.max(test_imgs)))

    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width,
                                                stride_height, stride_width)

    # print("\ntest PATCHES images shape:", patches_imgs_test.shape)
    # print("test PATCHES images range (min-max): " +
    #       str(np.min(patches_imgs_test)) + '-' +
    #       str(np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3]


def recompose(data, N_h, N_w):
    assert(len(data.shape) == 4)
    N_pacth_per_img = N_w * N_h
    assert(data.shape[0] % N_pacth_per_img == 0)
    N_full_imgs = data.shape[0] // N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]

    # define and start full recompose
    full_recomp = np.empty((N_full_imgs, data.shape[1], N_h*patch_h, N_w*patch_w))
    k = 0  # iter full img
    s = 0  # iter single patch
    while (s < data.shape[0]):
        # recompose one:
        single_recon = np.empty((data.shape[1], N_h*patch_h, N_w*patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:, h*patch_h:(h*patch_h)+patch_h, w*patch_w:(w*patch_w)+patch_w] = data[s]
                s += 1
        full_recomp[k] = single_recon
        k += 1
    assert(k == N_full_imgs)

    return full_recomp


def recompose_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)  #4D arrays
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    # print("N_patches_h: " +str(N_patches_h))
    # print("N_patches_w: " +str(N_patches_w))
    # print("N_patches_img: " +str(N_patches_img))
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    # print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    # print(final_avg.shape)
    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg





############################# Internal functions ##################################

# Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    # assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    N_patches_h = img_h // patch_h  # round to lowest int
    if img_h % patch_h != 0:
        print("warning: " + str(N_patches_h) + " patches in height, with about " + str(img_h%patch_h) + " pixels left over")
    N_patches_w = img_w // patch_w  # round to lowest int
    if img_h % patch_h != 0:
        print("warning: " + str(N_patches_w) + " patches in width, with about " + str(img_w%patch_w) + " pixels left over")
    # print("number of patches per image: " + str(N_patches_h * N_patches_w))
    N_patches_tot = (N_patches_h * N_patches_w) * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))

    iter_tot = 0   # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i, :, h*patch_h:(h*patch_h)+patch_h, w*patch_w:(w*patch_w)+patch_w]
                patches[iter_tot] = patch
                iter_tot += 1   # total
    assert(iter_tot == N_patches_tot)

    return patches, N_patches_h, N_patches_w   # array with all the full_imgs divided in patches


# Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    # assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)

    N_patches_h = ((img_h - patch_h) // stride_h + 1)
    N_patches_w = ((img_w - patch_w) // stride_w + 1)
    N_patches_img = N_patches_h * N_patches_w  # // division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]
    # print("Number of patches on h : ", ((img_h - patch_h) // stride_h + 1))
    # print("Number of patches on w : ", ((img_w-patch_w)//stride_w+1))
    # print("number of patches per image: " + str(N_patches_img) + ", totally for this dataset: " + str(N_patches_tot))

    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))
    iter_tot = 0   # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i, :, h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot] = patch
                iter_tot += 1   # total
    assert (iter_tot == N_patches_tot)

    return patches  # array with all the full_imgs divided in patches


def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    # assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image

    leftover_h = (img_h - patch_h) % stride_h  # leftover on the h dim
    leftover_w = (img_w - patch_w) % stride_w  # leftover on the w dim

    if leftover_h != 0:  # change dimension of img_h
        # print("\nThe side H is not compatible with the selected stride of " + str(stride_h))
        # print("img_h " + str(img_h) + ", patch_h " + str(patch_h) + ", stride_h " + str(stride_h))
        # print("(img_h - patch_h) MOD stride_h: " + str(leftover_h))
        # print("So the H dim will be padded with additional " + str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h+(stride_h-leftover_h), img_w))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs

    if leftover_w != 0:   # change dimension of img_w
        # print("The side W is not compatible with the selected stride of " + str(stride_w))
        # print("img_w " + str(img_w) + ", patch_w " + str(patch_w) + ", stride_w " + str(stride_w))
        # print("(img_w - patch_w) MOD stride_w: " + str(leftover_w))
        # print("So the W dim will be padded with additional " + str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2], 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    # print("new full images shape:", full_imgs.shape)

    return full_imgs


# Extend the full images becasue patch divison is not exact
def paint_border(data, patch_h, patch_w):
    assert (len(data.shape) == 4)  # 4D arrays
    # assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    img_h = data.shape[2]
    img_w = data.shape[3]
    if img_h % patch_h == 0:
        new_img_h = img_h
    else:
        new_img_h = ((img_h // patch_h) + 1) * patch_h
    if img_w % patch_w == 0:
        new_img_w = img_w
    else:
        new_img_w = ((img_w // patch_w) + 1) * patch_w
    new_data = np.zeros((data.shape[0], data.shape[1], new_img_h, new_img_w))
    new_data[:, :, 0:img_h, 0:img_w] = data[:, :, :, :]
    return new_data


