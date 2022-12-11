import shutil
import random
import os

from torchinfo import summary
import torchvision.utils as vutils
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
# import piqa
# import lpips

from AutoEncoder import AutoEncoder
from MyDataset import MyDataset
from MyLogger import get_logger

def ms(m):
    return summary(m, input_size=(1,3,180,180), col_names = [
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
        ])

def mix(a,b,m):
    return a + (b-a) * m

@th.no_grad()
def get_wd_params(model: nn.Module):
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():
        if hasattr(param,'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay

logger = get_logger('loss logs', logpath='logs.txt', displaying=False)
logger_norms = get_logger('norm logs', logpath='logs_norms.txt', displaying=False)

cuda = True
device = th.device("cuda:0" if cuda else "cpu")

#th.autograd.set_detect_anomaly(True)
save = True
load = False
load_path = 'weights/recent_ckpt.pth'

NUM_FIXED_IMAGES = 32

lr_start = .001
lr_end = .0001
MinLrAtEpoch = 1000
weight_decay = 0.000001
batch_size = NUM_FIXED_IMAGES
# _loss_func1 = lpips.LPIPS(net='vgg').to(device)
# _loss_func2 = piqa.ssim.MS_SSIM(window_size=8).to(device)
# _loss_func3 = nn.L1Loss()
_loss_func3 = nn.MSELoss()
def loss_func(yhat,y,dataset):
    reconstructions = yhat['reconstruction']
    # means = yhat['mean']
    # logvars = yhat['logvar']

    # recons_loss = _loss_func1(dataset.norm(reconstructions), dataset.norm(y)).mean()
    # recons_loss += 1 - _loss_func2(reconstructions, y)
    # recons_loss += _loss_func3(reconstructions, y)

    # recons_loss = _loss_func3(reconstructions, y)
    # kld_weight = batch_size / len(train_dataset)
    # kld_loss = th.mean(-.5 * th.sum(1 + logvars - means ** 2 - logvars.exp(), dim=1), dim=0)
    # return recons_loss + kld_weight * kld_loss

    # recons_loss = 4 / 5 * _loss_func1(th.clamp(reconstructions,0,1) * 2 - 1, y * 2 - 1).mean()
    # recons_loss += 1 / 5 * _loss_func3(reconstructions, y)

    # recons_loss = 1 / 5 * _loss_func1(th.clamp(reconstructions,0,1) * 2 - 1, y * 2 - 1).mean()
    # recons_loss += _loss_func3(reconstructions, y)

    # # recons_loss = 4 / 5 * (1-_loss_func2(th.clamp(reconstructions,0,1),y).mean())
    # recons_loss = 4 / 5 * (1-_loss_func2(reconstructions,y).mean())
    # recons_loss += 1 / 5 * _loss_func3(reconstructions, y)

    # recons_loss = .25 * (1 - _loss_func2(th.clamp(reconstructions,0,1), y))
    # recons_loss += .75 * _loss_func3(reconstructions, y)

    recons_loss = _loss_func3(reconstructions, y)
    return recons_loss

model = AutoEncoder().to(device)
decay_params, no_decay_params = get_wd_params(model)
# TODO try adamW?
optimizer = optim.Adam([{'params': no_decay_params, 'weight_decay': 0}, {'params': decay_params, 'weight_decay': 0}], lr=lr_start)
scheduler = CosineAnnealingLR(optimizer, MinLrAtEpoch, eta_min=lr_end)

# ms(model)

best_avg_loss = float('inf')
best_test_loss = float('inf')
avg_loss = 0
ema_loss = 1
epoch = 0

dataset = MyDataset('../dataset.pkl')
fixed_images_indices = []

total_params = sum(param.numel() for param in model.parameters())

if load:
    if not os.path.exists(load_path):
        print('Checkpoint not found at path:',load_path)
        print('Exiting!')
        exit()
    ckpt = th.load(load_path)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])

    dataset.load_indices(ckpt['dataset_indices']) # maintain same train / test split
    epoch = ckpt['epoch']
    best_avg_loss = ckpt['best_avg_loss']
    avg_loss = ckpt['avg_loss']
    ema_loss = avg_loss

    fixed_images_indices = ckpt['fixed_images_indices']

    print('Loaded checkpoint:')
    print('\tlr:',optimizer.param_groups[0]['lr'])
    print('\tepoch:',epoch)
    print('\tavg_loss    :',avg_loss)
    print('\tbest_avg_loss:',best_avg_loss)
elif save:
    logger.info("Number of parameters:" + str(total_params))

print("Number of parameters:" + str(total_params))

train_dataset, test_dataset = dataset.get_train_test_subsets()
params = {
    "batch_size":batch_size,
    "shuffle":True,
    "num_workers":4
}
train_generator = th.utils.data.DataLoader(train_dataset, **params)
test_generator = th.utils.data.DataLoader(test_dataset, **params)


if len(fixed_images_indices) == 0:
    fixed_images_indices = list(range(len(test_dataset)))
    random.shuffle(fixed_images_indices)
    fixed_images_indices = fixed_images_indices[:NUM_FIXED_IMAGES]
fixed_images = [test_dataset[i][0].unsqueeze(0) for i in fixed_images_indices]
fixed_states = [test_dataset[i][1].unsqueeze(0) for i in fixed_images_indices]
fixed_images = th.cat(fixed_images).to(device)
fixed_states = th.cat(fixed_states).to(device)
# vutils.save_image(dataset.unNormalize(fixed_images), 'outputs/original.png')
vutils.save_image(fixed_images, 'outputs/original.png')
vutils.save_image(model(fixed_images, fixed_states)['reconstruction'], 'outputs/recent.png')
# vutils.save_image(model.sample(64), 'outputs/sample.png')

while True:
    progress_bar = tqdm(range((len(train_dataset) + batch_size - 1)//batch_size))

    loss_sum = 0
    num_processed = 0
    for i,(X,S) in zip(progress_bar,train_generator):
        X_batch_size = X.shape[0]

        X = X.to(device)
        Y = X
        S = S.to(device) # extra info injected into the autoencoder

        Y_hat = model(X,S)

        try:
            loss = loss_func(Y_hat,Y,dataset) 
        except Exception as e:
            print(e)
            optimizer.zero_grad()
            print('Exception thrown when computing loss. Possible NaNs?')
            continue
        if loss.item() > 10.0 or loss.item() < 0:
            optimizer.zero_grad()
            print('Loss too big:',loss.item())
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item() * X_batch_size
        ema_loss = mix(ema_loss, loss.item(), 10*batch_size/len(train_dataset))
        num_processed += X_batch_size

        num_processed = 1 if num_processed == 0 else num_processed
        info = 'Epoch:{} avgLoss:{:9.7f} Loss:{:9.7f} Lr:{:9.7f}'.format(epoch, loss_sum / num_processed, ema_loss, optimizer.param_groups[0]['lr'])
        progress_bar.set_description(info)

    with th.no_grad():
        model.eval()

        epoch += 1
        if epoch <= MinLrAtEpoch:
            scheduler.step()

        avg_loss = loss_sum / num_processed
        prev_best_avg_loss = best_avg_loss
        best_avg_loss = min(best_avg_loss, avg_loss)

        if save:
            ckpt = {
                    'model_state':model.state_dict(),
                    'optimizer_state':optimizer.state_dict(),
                    'scheduler_state':scheduler.state_dict(),
                    'dataset_indices':dataset.indices,
                    'epoch':epoch,
                    'best_avg_loss':best_avg_loss,
                    'avg_loss':avg_loss,
                    'best_test_loss':best_test_loss,

                    'fixed_images_indices':fixed_images_indices,
                }
            th.save(ckpt, 'weights/recent_ckpt.pth')
            if avg_loss < prev_best_avg_loss:
                shutil.copyfile('weights/recent_ckpt.pth','weights/best_train_loss_ckpt.pth',follow_symlinks=False)

        # calculate test loss
        if epoch % 2 == 0:
            loss_sum = 0
            num_processed = 0
            for X,S in test_generator:
                X_batch_size = X.shape[0]
                num_processed += X_batch_size

                X = X.to(device)
                Y = X
                S = S.to(device)

                try:
                    loss = loss_func(model(X,S),Y,dataset) 
                except:
                    print('Exception thrown when computing loss')
                    loss_sum += 1 * X_batch_size
                    continue
                if loss.item() > 2.0 or loss.item() < 0:
                    print('Loss too big:',loss.item())
                    loss_sum += 1 * X_batch_size
                    continue
                loss_sum += loss.item() * X_batch_size

            avg_loss = loss_sum / num_processed
            info += '\nTest loss: {:9.7f}\n'.format(avg_loss)
            if avg_loss < best_test_loss:
                best_test_loss = avg_loss
                shutil.copyfile('weights/recent_ckpt.pth','weights/best_test_loss_ckpt.pth',follow_symlinks=False)

        # store outputs
        Y_hat = model(fixed_images,fixed_states)['reconstruction']
        vutils.save_image(Y_hat, 'outputs/recent.png')
        if epoch % 2 == 0:
            vutils.save_image(Y_hat,  'outputs/' + str(epoch) + '.png')

        # Y_hat = model.sample(64)
        # vutils.save_image(Y_hat, 'outputs/sample.png')

        if save:
            logger.info(info)
        else:
            print(info)

        model.train()

# mean = 0.
# std = 0.
# low_x = 1111
# high_x = -1111
# low_y = 1111
# high_y = -1111
# for x,y in train_generator:
#     batch_samples = x.size(0) # batch size (the last batch can have smaller size!)
#     x = x.view(batch_samples, x.size(1), -1)
#     mean += x.mean(2).sum(0)
#     std += x.std(2).sum(0)
#     low_x = min(x.min(),low_x)
#     high_x = max(x.max(),high_x)
#     low_y = min(y.min(),low_y)
#     high_y = max(y.max(),high_y)
#
# mean /= len(train_generator.dataset)
# std /= len(train_generator.dataset)
# print('mean:',mean)
# print('std:',std)
# print('min_y:',low_y)
# print('max_y:',high_y)
# print('min_x:',low_x)
# print('max_x:',high_x)
# exit()

