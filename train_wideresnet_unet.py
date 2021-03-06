from sklearn.model_selection import train_test_split

from common import *
from consts import *
from utils import init
from utils.mo_dataset import MODataset
from utils.metrics import dice_value, MetricMonitor, criterion_BCE_SoftDice
from utils import augmentation
from models.wideresnet_unet.ternausnet2 import TernausNetV2

init.set_results_reproducible()
init.init_torch()


############################# Load Data ##################################
def train_transforms(image, masks, labels=None):
    seq = augmentation.get_train_augmenters_seq2(mode='constant')
    hooks_masks = augmentation.get_train_masks_augmenters_deactivator()

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
    image = seq_det.augment_images([image])[0]
    image = transforms.ToTensor()(image.copy())
    image = transforms.Normalize(IMAGES_MEAN, IMAGES_STD)(image)

    masks = seq_det.augment_images([masks], hooks=hooks_masks)[0]
    masks = (masks >= MASK_THRESHOLD).astype(np.uint8)

    if labels is not None:
        labels = seq_det.augment_images([labels], hooks=hooks_masks)[0]
        labels = (labels > 0).astype(np.uint8)
        # for index in range(labels.shape[-1]):
        #     labels[..., index] = (labels[..., index] > 0).astype(np.uint8)
        return image, masks, labels
    else:
        return image, masks


def valid_transforms(image, masks, labels=None):
    image = transforms.ToTensor()(image.copy())
    image = transforms.Normalize(IMAGES_MEAN, IMAGES_STD)(image)
    if labels is not None:
        return image, masks, labels
    else:
        return image, masks


trans = {'train': train_transforms, 'valid': valid_transforms}
all_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
train_ids = [i for i in all_ids if i not in TEST_IDS]
ids_train, ids_valid = train_test_split(train_ids, test_size=0.2, random_state=42)
ids = {'train': ids_train, 'valid': ids_valid}
# ids = {'train': R_TRAIN_IDS, 'valid': R_VALID_IDS}
datasets = {x: MODataset(INPUT_DIR,
                         ids[x],
                         num_patches=100,
                         patch_size=128,
                         masks=['touching', 'inside'],
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
                log_save_path=None, num_epochs=25, iter_size=1, compare_Loss=False):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val = -sys.maxsize
    monitor = MetricMonitor()
    log = open(log_save_path, 'a') if log_save_path is not None else \
          type('dummy', (object,), {'write': lambda x,y:0, 'flush': lambda x:0, 'close': lambda x:0})()
    log.write(f'Training start at {time.strftime("%Y-%m-%d %H:%M")}\n\n')

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
                # get the weight map
                # weight = torch.tensor(samples['weight'], dtype=torch.float).cuda(async=True)

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
                dice = dice_value(outputs.data, targets.data, [0, 1])
                monitor.update('loss', loss.data, inputs.shape[0])
                monitor.update('dice', dice.data, inputs.shape[0])
                stream.set_description(f'epoch {epoch+1}/{num_epochs} | {phase}: {monitor}')
            stream.close()

            epoch_loss = monitor.get_avg('loss')
            epoch_dice = monitor.get_avg('dice')
            epoch_val = epoch_dice if not compare_Loss else -epoch_loss

            log.write(f'epoch {epoch+1}/{num_epochs} | {phase}: {monitor} | lr {optimizer.param_groups[0]["lr"]:.0e}\n')

            if phase == 'valid' and scheduler is not None:
                scheduler.step(-epoch_val)

            # save the model and optimizer
            if (phase == 'valid') and (epoch_val > best_val):
                best_val = epoch_val
                best_model_wts = copy.deepcopy(model.state_dict())
                if model_save_path is not None:
                    path = model_save_path.format((epoch+1), abs(best_val))
                    torch.save(best_model_wts, path)
                    print(f'Weights of model saved at {path}')
                    log.write(f'Weights of model saved at {path}\n')
            if (phase == 'valid') and (optim_save_path is not None):
                path = optim_save_path.format((epoch + 1), optimizer.param_groups[0]['lr'])
                torch.save(optimizer.state_dict(), path)
            log.flush()

        log.write('\n')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed//60):.0f}m {(time_elapsed%60):.0f}s')
    print(f'Best Val: {best_val:.4f}')
    log.write(f'Training complete in {(time_elapsed//60):.0f}m {(time_elapsed%60):.0f}s\n')
    log.write(f'Best Val: {best_val:f}\n')
    log.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


########################### Config Train ##############################
net = TernausNetV2(num_classes=2, num_input_channels=3).cuda()
# weight_path = os.path.join(WEIGHTS_DIR, 'deepglobe_buildings.pt')
# state = torch.load(weight_path)
# state = {key.replace('module.', ''): value for key, value in state['model'].items()}
# net.load_state_dict(state)


def criterion(outputs, masks):
    return criterion_BCE_SoftDice(outputs, masks, dice_w=None)


print('\n---------------- Training unet ----------------')
optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=5e-3, momentum=0.9, weight_decay=0.0001)

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

model_save_path = os.path.join(WEIGHTS_DIR, 'final2/unet_{:d}_{:.4f}.pth')
optim_save_path = os.path.join(WEIGHTS_DIR, 'final2/optim.pth')
log_save_path = os.path.join(WEIGHTS_DIR, 'final2/log.txt')
net = train_model(net, criterion, optimizer, exp_lr_scheduler, model_save_path, optim_save_path,
                  log_save_path, num_epochs=50)

