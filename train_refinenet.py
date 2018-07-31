from sklearn.model_selection import train_test_split

from common import *
from consts import *
from utils import init
from utils.mo_dataset import MODataset
from utils.metrics import criterion_BCE_SoftDice, dice_value, MetricMonitor
from utils import augmentation
from models.refinenet.refinenet_4cascade import RefineNet4Cascade

init.set_results_reproducible()
init.init_torch()


############################# Load Data ##################################
def train_transforms(image, masks):
    seq = augmentation.get_train_augmenters_seq()
    hooks_masks = augmentation.get_train_masks_augmenters_deactivator()

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
    image_aug = seq_det.augment_images([image])[0]
    masks_aug = seq_det.augment_images([masks], hooks=hooks_masks)[0]

    image_aug_tensor = transforms.ToTensor()(image_aug.copy())
    image_aug_tensor = transforms.Normalize([0.8275685641750257, 0.5215321518722066, 0.646311050624383],
                                            [0.16204139441725898, 0.248547854527502, 0.2014914668413328])(image_aug_tensor)

    masks_aug = (masks_aug >= MASK_THRESHOLD).astype(np.uint8)

    return image_aug_tensor, masks_aug


def valid_transforms(image, masks):
    img_tensor = transforms.ToTensor()(image.copy())
    img_tensor = transforms.Normalize([0.8275685641750257, 0.5215321518722066, 0.646311050624383],
                                       [0.16204139441725898, 0.248547854527502, 0.2014914668413328])(img_tensor)

    return img_tensor, masks


trans = {'train': train_transforms, 'valid': valid_transforms}
all_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
train_ids = [i for i in all_ids if i not in TEST_IDS]
ids_train, ids_valid = train_test_split(train_ids, test_size=0.2, random_state=42)
ids = {'train': ids_train, 'valid': ids_valid}
datasets = {x: MODataset(INPUT_DIR,
                         ids[x],
                         num_patches=200,
                         patch_size=256,
                         transform=trans[x])
           for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                              batch_size=8,
                                              shuffle=True, 
                                              num_workers=8,
                                              pin_memory=True)
              for x in ['train', 'valid']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'valid']}


############################# Training the model ##################################
def train_model(model, criterion, optimizer, scheduler=None, save_path=None, num_epochs=25, iter_size=1):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = 0
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
                targets = torch.tensor(samples['masks'], dtype=torch.float).cuda(async=True)

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
                dice = dice_value(outputs.data, targets.data, [0.5, 0.5, 0])
                monitor.update('loss', loss.data, inputs.shape[0])
                monitor.update('dice', dice.data, inputs.shape[0])
                stream.set_description(
                    f'epoch {epoch+1}/{num_epochs} | '
                    f'{phase}: {monitor}'
                )
            stream.close()

            epoch_loss = monitor.get_avg('loss')
            epoch_dice = monitor.get_avg('dice')

            if phase == 'valid' and scheduler is not None:
                scheduler.step(epoch_dice)
            
            # deep copy the model
            if (phase == 'valid') and (epoch_dice > best_dice):
                best_dice = epoch_dice
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_path is not None:
                    path = save_path.format(best_dice)
                    torch.save(best_model_wts, path)
                    print('Weights of model saved at {}'.format(path))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Dice: {:.4f}'.format(best_dice))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


########################### Config Train ##############################

net = RefineNet4Cascade((3, 128), num_classes=1, pretrained=False, freeze_resnet=False).cuda()
# layers = [net.layer1_rn, net.layer2_rn, net.layer3_rn, net.layer4_rn, net.refinenet1, net.refinenet2,
#           net.refinenet3, net.refinenet4, net.output_conv]
# for layer in layers:
#     for param in layer.parameters():
#         param.requires_grad = True


def criterion(logits, labels):
    return criterion_BCE_SoftDice(logits, labels, dice_w=[0.5, 0.5, 0], use_weight=False)


optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

save_path = os.path.join(WEIGHTS_DIR, 'REFINENET', 'refinenet-{:.4f}.pth')
net = train_model(net, criterion, optimizer, exp_lr_scheduler, save_path, num_epochs=30)

