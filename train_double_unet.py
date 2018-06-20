from sklearn.model_selection import train_test_split

from common import *
from consts import *
from utils import init
from utils.mo_dataset_double import MODatasetDouble
from utils.metrics import criterion_BCE_SoftDice, criterion_AngularError, dice_value
from models.double_unet import DoubleUNet

init.set_results_reproducible()
init.init_torch()

############################# Load Data ##################################

def train_transforms(image, masks):
    image = transforms.ToTensor()(image)
    return image, masks

def valid_transforms(image, masks):
    image = transforms.ToTensor()(image)
    return image, masks

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
                                              batch_size=8,
                                              shuffle=True, 
                                              num_workers=8,
                                              pin_memory=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}



############################# Training the model ################################## 

def train_model(model, criterion1, criterion2, optimizer, scheduler = None, save_path = None, num_epochs = 25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = 0

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
                loss1 = criterion1(outputs1, targets1, areas)
                loss2 = criterion2(outputs2, targets2)
                loss = loss1 + loss2

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data * inputs.shape[0]
                running_dice += (dice_value(outputs2.data[:, 0], targets2.data[:, 0]) +
                                 dice_value(outputs2.data[:, 1], targets2.data[:, 1])) * inputs.shape[0]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_dice = running_dice / dataset_sizes[phase]

            print('{} Loss: {:.4f} Dice: {:.4f}'.format(phase, epoch_loss, epoch_dice))
            if phase == 'val' and scheduler is not None:
                scheduler.step(epoch_dice)
            
            # deep copy the model
            if (phase == 'val') and (epoch_dice > best_dice):
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
    print('Best val Dice: {:4f}'.format(best_dice))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


########################### Config Train ##############################

net = DoubleUNet(DOUBLE_UNET_CONFIG).cuda()

# Observe that all parameters are being optimized
optimizer = optim.SGD(filter(lambda p:  p.requires_grad, net.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

save_path = os.path.join(WEIGHTS_DIR, 'double-unet-{:.4f}.pth')
model = train_model(net, criterion_AngularError, criterion_BCE_SoftDice, optimizer, exp_lr_scheduler,
                    save_path, num_epochs=5)
