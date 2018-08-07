from sklearn.model_selection import train_test_split

from common import *
from consts import *
from utils import init
from utils.mo_dataset_d import MODatasetD
from utils.metrics import criterion_CCE_SoftDice, ce_dice_value, MetricMonitor
from utils import augmentation
from models.vgg_unet import VGG_UNet16

init.set_results_reproducible()
init.init_torch()


############################# Load Data ##################################
def train_transforms(image, masks, labels=None):
    seq = augmentation.get_train_augmenters_seq2(mode='constant')
    hooks_masks = augmentation.get_train_masks_augmenters_deactivator()

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
    image_aug = seq_det.augment_images([image])[0]
    image_aug_tensor = transforms.ToTensor()(image_aug.copy())
    image_aug_tensor = transforms.Normalize(IMAGES_MEAN, IMAGES_STD)(image_aug_tensor)

    masks_aug = seq_det.augment_images([masks], hooks=hooks_masks)[0]
    masks_aug = (masks_aug >= MASK_THRESHOLD).astype(np.uint8)

    if labels is not None:
        labels_aug = seq_det.augment_images([labels], hooks=hooks_masks)[0]
        for index in range(labels_aug.shape[-1]):
            labels_aug[..., index] = (labels_aug[..., index] > 0).astype(np.uint8)
        return image_aug_tensor, masks_aug, labels_aug
    else:
        return image_aug_tensor, masks_aug


def valid_transforms(image, masks, labels=None):
    img_tensor = transforms.ToTensor()(image.copy())
    img_tensor = transforms.Normalize(IMAGES_MEAN, IMAGES_STD)(img_tensor)
    if labels is not None:
        return img_tensor, masks, labels
    else:
        return img_tensor, masks


trans = {'train': train_transforms, 'valid': valid_transforms}
all_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
train_ids = [i for i in all_ids if i not in TEST_IDS]
ids_train, ids_valid = train_test_split(train_ids, test_size=0.2, random_state=42)
ids = {'train': ids_train, 'valid': ids_valid}
datasets = {x: MODatasetD(INPUT_DIR,
                          ids[x],
                          num_patches=100,
                          patch_size=128,
                          masks=['touching', 'inside'],
                          centroid_size=5,
                          transform=trans[x])
            for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                              batch_size=4,
                                              shuffle=True, 
                                              num_workers=8,
                                              pin_memory=True)
               for x in ['train', 'valid']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'valid']}


############################# Training the model ##################################
def train_model(model, criterion, optimizer, scheduler=None, model_save_path=None, optim_save_path=None,
                num_epochs=25, iter_size=1, compare_Loss=False):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val = -sys.maxsize
    monitor = MetricMonitor()

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            model.train(phase == 'train')  # Set model to training/evaluate mode
            optimizer.zero_grad()
            monitor.reset()
            stream = tqdm(dataloaders[phase], file=sys.stdout)
            # Iterate over data.
            for i, samples in enumerate(stream, start=1):
                # get the inputs
                inputs = torch.tensor(samples['image'], requires_grad=True).cuda(async=True)
                # get the targets
                masks = torch.tensor(samples['masks'], dtype=torch.long).cuda(async=True)
                centroids = torch.tensor(samples['centroids'], dtype=torch.long).cuda(async=True)
                targets = masks + centroids

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    if i % iter_size == 0 or i == len(dataloaders[phase]):
                        optimizer.step()
                        optimizer.zero_grad()

                # statistics
                dice = ce_dice_value(outputs.data, targets.data, [0, 0, 1, 0])
                monitor.update('loss', loss.data, inputs.shape[0])
                monitor.update('dice', dice.data, inputs.shape[0])
                stream.set_description(
                    f'epoch {epoch+1}/{num_epochs} | '
                    f'{phase}: {monitor}'
                )
            stream.close()

            epoch_loss = monitor.get_avg('loss')
            epoch_dice = monitor.get_avg('dice')
            epoch_val = epoch_dice if not compare_Loss else -epoch_loss

            if phase == 'valid' and scheduler is not None:
                scheduler.step(-epoch_val)
            
            # deep copy the model
            if (phase == 'valid') and (epoch_val > best_val):
                best_val = epoch_val
                best_model_wts = copy.deepcopy(model.state_dict())
                if model_save_path is not None:
                    path = model_save_path.format((epoch+1), abs(best_val))
                    torch.save(best_model_wts, path)
                    print('Weights of model saved at {}'.format(path))
            if (phase == 'valid') and (optim_save_path is not None):
                path = optim_save_path.format((epoch + 1), optimizer.param_groups[0]['lr'])
                torch.save(optimizer.state_dict(), path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Dice: {:.4f}'.format(best_val))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


########################### Config Train ##############################
net = VGG_UNet16(num_classes=4, pretrained=True).cuda()

# weight_path = os.path.join(WEIGHTS_DIR, 'UNET3/unet-0.5996.pth')
# net.load_state_dict(torch.load(weight_path))

def criterion(outputs, masks):
    return criterion_CCE_SoftDice(outputs, masks,
                                  dice_w=[0.1, 0.3, 0.2, 0.4],
                                  ce_w  =[0.1, 0.3, 0.2, 0.4])

print('\n---------------- Training unet ----------------')
# optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=5e-3,
#                       momentum=0.9, weight_decay=0.0001)
optimizer = optim.Adam(filter(lambda p:  p.requires_grad, net.parameters()), lr=1e-3)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

model_save_path = os.path.join(WEIGHTS_DIR, 'test/unet_{:d}_{:.4f}.pth')
optim_save_path = os.path.join(WEIGHTS_DIR, 'test/optim.pth')
net = train_model(net, criterion, optimizer, exp_lr_scheduler, model_save_path, optim_save_path, num_epochs=40)
