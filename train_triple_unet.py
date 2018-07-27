from sklearn.model_selection import train_test_split

from common import *
from consts import *
from utils import init
from utils.mo_dataset_double import MODatasetDouble
from utils.metrics import criterion_BCE_SoftDice, criterion_AngularError, criterion_MSELoss, dice_value, MetricMonitor
from utils import augmentation
from models.unet import TripleUNet, TripleWiredUNet

init.set_results_reproducible()
init.init_torch()


############################# Load Data ##################################
def train_transforms(image, mask, labels):
    seq = augmentation.get_train_augmenters_seq()
    hooks_masks = augmentation.get_train_masks_augmenters_deactivator()

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
    image_aug = seq_det.augment_images([image])[0]
    mask_aug = seq_det.augment_images([mask], hooks=hooks_masks)[0]
    labels_aug = seq_det.augment_images([labels], hooks=hooks_masks)[0]

    mask_aug = (mask_aug >= MASK_THRESHOLD).astype(np.uint8)
    for index in range(labels_aug.shape[-1]):
        labels_aug[..., index] = (labels_aug[..., index] > 0).astype(np.uint8)

    image_aug_tensor = transforms.ToTensor()(image_aug.copy())
    # image_aug_tensor = transforms.Normalize([0.03072981, 0.03072981, 0.01682784],
    #                              [0.17293351, 0.12542403, 0.0771413 ])(image_aug_tensor)

    return image_aug_tensor, mask_aug, labels_aug


def valid_transforms(image, mask, labels):
    img_tensor = transforms.ToTensor()(image.copy())
    # img_tensor = transforms.Normalize([0.03072981, 0.03072981, 0.01682784],
    #                              [0.17293351, 0.12542403, 0.0771413 ])(img_tensor)

    return img_tensor, mask, labels


trans = {'train': train_transforms, 'valid': valid_transforms}
all_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
train_ids = [i for i in all_ids if i not in TEST_IDS]
ids_train, ids_valid = train_test_split(train_ids, test_size=0.2, random_state=42)
# ids_train = [i for i in all_ids if i not in TEST_IDS]
# ids_valid = TEST_IDS
ids = {'train': ids_train, 'valid': ids_valid}
datasets = {x: MODatasetDouble(INPUT_DIR,
                               ids[x],
                               num_patches=100,
                               patch_size=128,
                               transform=trans[x],
                               erosion=1,
                               gate_image=False)
           for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                              batch_size=4,
                                              shuffle=True, 
                                              num_workers=8,
                                              pin_memory=True)
              for x in ['train', 'valid']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'valid']}


############################# Training the model ##################################
def train_model(model, criterion1, criterion2, criterion3, optimizer, scheduler=None, save_path=None,
                num_epochs=25, iter_size=1, compare='loss'):
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
                vectors = torch.tensor(samples['vectors'], dtype=torch.float).cuda(async=True)
                masks = torch.tensor(samples['masks'], dtype=torch.float).cuda(async=True)
                areas = torch.tensor(samples['areas'], dtype=torch.float).cuda(async=True)

                # forward
                outputs1, outputs2, outputs3 = model(inputs)
                loss1 = criterion1(inputs, outputs1, vectors, masks, areas) if criterion1 is not None else 0
                loss2 = criterion2(inputs, outputs2, vectors, masks, areas) if criterion2 is not None else 0
                loss3 = criterion3(inputs, outputs3, vectors, masks, areas) if criterion3 is not None else 0
                loss = loss1 + loss2 + loss3

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    if i % iter_size == 0 or i == len(dataloaders[phase]):
                        optimizer.step()
                        optimizer.zero_grad()

                # statistics
                dice1 = dice_value(outputs1.data, torch.unsqueeze(masks[:, 0], 1).data)
                dice3 = dice_value(outputs3.data, masks.data)
                monitor.update('loss', loss.data, inputs.shape[0])
                monitor.update('dice1', dice1.data, inputs.shape[0])
                monitor.update('dice3', dice3.data, inputs.shape[0])
                stream.set_description(
                    f'epoch {epoch+1}/{num_epochs} | '
                    f'{phase}: {monitor}'
                )
            stream.close()

            epoch_val = monitor.get_avg('dice1') if compare == 'dice1' else \
                       (monitor.get_avg('dice3') if compare == 'dice3' else -monitor.get_avg('loss'))

            if phase == 'valid' and scheduler is not None:
                scheduler.step(epoch_val)
            
            # deep copy the model
            if (phase == 'valid') and (epoch_val > best_val):
                best_val = epoch_val
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_path is not None:
                    path = save_path.format((epoch+1), optimizer.param_groups[0]['lr'], abs(best_val))
                    torch.save(best_model_wts, path)
                    print('Weights of model saved at {}'.format(path))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Val: {:.4f}'.format(abs(best_val)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


########################### Config Train ##############################

net = TripleWiredUNet(TRIPLE_UNET_CONFIG_1).cuda()

def criterion1(inputs, outputs, vectors, masks, areas):
    return criterion_BCE_SoftDice(outputs, torch.unsqueeze(masks[:, 0], 1), use_weight=False)

def criterion2(inputs, outputs, vectors, masks, areas):
    return criterion_AngularError(outputs, vectors, areas)
    # return criterion_MSELoss(outputs, vectors)

def criterion3(inputs, outputs, vectors, masks, areas):
    return criterion_BCE_SoftDice(outputs, masks, dice_w=[0.3, 0.7], use_weight=False)


# weight_path = os.path.join(WEIGHTS_DIR, 'final/dwunet1_20_1e-04_10.6428.pth')
# net.load_state_dict(torch.load(weight_path))
print('\n---------------- Training first unet ----------------')
for param in net.unet1.parameters():
    param.requires_grad = True
for param in net.unet2.parameters():
    param.requires_grad = False
for param in net.unet3.parameters():
    param.requires_grad = False
save_path = os.path.join(WEIGHTS_DIR, 'test/tunet1_{:d}_{:.0e}_{:.4f}.pth')
optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
net = train_model(net, criterion1, None, None, optimizer, exp_lr_scheduler, save_path, num_epochs=30)

print('\n---------------- Training second unet ----------------')
for param in net.unet1.parameters():
    param.requires_grad = False
for param in net.unet2.parameters():
    param.requires_grad = True
for param in net.unet3.parameters():
    param.requires_grad = False
save_path = os.path.join(WEIGHTS_DIR, 'test/tunet2_{:d}_{:.0e}_{:.4f}.pth')
optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
net = train_model(net, None, criterion2, None, optimizer, exp_lr_scheduler, save_path, num_epochs=10)

print('\n---------------- Training third unet ----------------')
for param in net.unet1.parameters():
    param.requires_grad = False
for param in net.unet2.parameters():
    param.requires_grad = False
for param in net.unet3.parameters():
    param.requires_grad = True
save_path = os.path.join(WEIGHTS_DIR, 'test/tunet3_{:d}_{:.0e}_{:.4f}.pth')
optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
net = train_model(net, None, criterion2, None, optimizer, exp_lr_scheduler, save_path, num_epochs=10)

print('\n---------------- Fine-tuning entire net ----------------')
for param in net.unet1.parameters():
    param.requires_grad = True
for param in net.unet2.parameters():
    param.requires_grad = True
for param in net.unet3.parameters():
    param.requires_grad = True
save_path = os.path.join(WEIGHTS_DIR, 'test/tunet_all_{:d}_{:.0e}_{:.4f}.pth')
optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
net = train_model(net, criterion1, criterion2, criterion3, optimizer, exp_lr_scheduler, save_path, num_epochs=20)
