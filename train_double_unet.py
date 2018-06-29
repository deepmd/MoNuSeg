from sklearn.model_selection import train_test_split

from common import *
from consts import *
from utils import init
from utils.mo_dataset_double import MODatasetDouble
from utils.metrics import criterion_BCE_SoftDice, criterion_AngularError, dice_value
from utils import augmentation
from models.double_unet import DoubleUNet

init.set_results_reproducible()
init.init_torch()

############################# Load Data ##################################
def train_transforms(image, masks):
    seq = augmentation.get_train_augmenters_seq()
    hooks_masks = augmentation.get_train_masks_augmenters_deactivator()

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
    image_aug = seq_det.augment_images([image])
    masks_aug = seq_det.augment_images([masks], hooks=hooks_masks)[0]

    image_aug_tensor = transforms.ToTensor()(image_aug[0].copy())
    # image_aug_tensor = transforms.Normalize([0.03072981, 0.03072981, 0.01682784],
    #                              [0.17293351, 0.12542403, 0.0771413 ])(image_aug_tensor)

    masks_aug[:, :, :-1] = (masks_aug[:, :, :-1] >= MASK_THRESHOLD).astype(np.uint8)

    return image_aug_tensor, masks_aug


def valid_transforms(image, masks):
    img_tensor = transforms.ToTensor()(image.copy())
    # img_tensor = transforms.Normalize([0.03072981, 0.03072981, 0.01682784],
    #                              [0.17293351, 0.12542403, 0.0771413 ])(img_tensor)

    return img_tensor, masks


trans = {'train': train_transforms, 'val': valid_transforms}
all_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPUT_DIR, IMAGES_DIR))]
train_ids = [i for i in all_ids if i not in TEST_IDS]
ids_train, ids_valid = train_test_split(train_ids, test_size=0.2, random_state=42)
ids = {'train': ids_train, 'val': ids_valid}
datasets = {x: MODatasetDouble(INPUT_DIR,
                               ids[x],
                               num_patches=20,
                               patch_size=256,
                               transform=trans[x])
           for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                              batch_size=4,
                                              shuffle=True, 
                                              num_workers=8,
                                              pin_memory=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}



############################# Training the model ################################## 

def train_model(model, criterion1, criterion2, optimizer, scheduler = None, save_path = None, num_epochs = 25, compare_Loss=False):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_dice = 0.0
                        
            # Iterate over data.
            for samples in dataloaders[phase]:
                # get the inputs
                inputs = torch.tensor(samples['image'], requires_grad=True).cuda(async=True)
                # get the targets
                targets1 = torch.tensor(samples['vectors'], dtype=torch.float).cuda(async=True)
                targets2 = torch.tensor(samples['masks'], dtype=torch.float).cuda(async=True)
                areas = torch.tensor(samples['areas'], dtype=torch.float).cuda(async=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs1, outputs2 = model(inputs)
                loss1 = criterion1(outputs1, targets1, areas) if criterion1 is not None else 0
                loss2 = criterion2(outputs2, targets2) if criterion2 is not None else 0
                loss = loss1 + loss2

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data * inputs.shape[0]
                running_dice += dice_value(outputs2.data, targets2.data, [0.3, 0.7]) * inputs.shape[0]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_dice = running_dice / dataset_sizes[phase]
            epoch_val = epoch_dice if not compare_Loss else (sys.maxsize - epoch_loss.data.cpu().numpy())

            print('{} Loss: {:.4f} Dice: {:.4f}'.format(phase, epoch_loss, epoch_dice))
            if phase == 'val' and scheduler is not None:
                scheduler.step(epoch_val)
            
            # deep copy the model
            if (phase == 'val') and (epoch_val > best_val):
                best_val = epoch_val
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_path is not None:
                    path = save_path.format(best_val)
                    torch.save(best_model_wts, path)
                    print('Weights of model saved at {}'.format(path))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Val: {:4f}'.format(best_val))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


########################### Config Train ##############################

net = DoubleUNet(DOUBLE_UNET_CONFIG).cuda()

def criterion1(logits, labels, areas):
    return criterion_AngularError(logits, labels, weights=areas)

def criterion2(logits, labels):
    return criterion_BCE_SoftDice(logits, labels, dice_w=[0.3, 0.7], use_weight=False)

print('\n---------------- Training first unet ----------------')
for param in net.unet2.parameters():
    param.requires_grad = False
optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
net = train_model(net, criterion1, None, optimizer, exp_lr_scheduler, None, num_epochs=15, compare_Loss=True)

print('\n---------------- Training second unet ----------------')
for param in net.unet1.parameters():
    param.requires_grad = False
for param in net.unet2.parameters():
    param.requires_grad = True
optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
net = train_model(net, None, criterion2, optimizer, exp_lr_scheduler, None, num_epochs=15)

print('\n---------------- Fine-tuning entire net ----------------')
for param in net.unet1.parameters():
    param.requires_grad = True
save_path = os.path.join(WEIGHTS_DIR, 'double-unet-{:.4f}.pth')
optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=0.0001,
                      momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
net = train_model(net, criterion1, criterion2, optimizer, exp_lr_scheduler, save_path, num_epochs=25)
