import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch

from torch import optim
from torch.utils.data import DataLoader
from utils.solvers import PolyLR
from utils.loss import HDRLoss
from utils.HDRutils import tonemap
from utils.dataprocessor import dump_sample
from dataset.HDR import KalantariDataset, KalantariTestDataset
from dataset_samsung_preload import Samsung_Dataset, Samsung_Dataset_test
from models.DeepHDR import DeepHDR
from utils.configs import Configs
import random
import numpy as np
import torchvision as tv

print(torch.cuda.is_available())

def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * np.log10(((1.0)*(1.0))/sqrdErr)

# Get configurations
configs = Configs()

# Load Data & build dataset
train_dataset = Samsung_Dataset("/data2/jaep0805/datasets/samsungdataset/CVPR2020_NTIRE_Workshop/train")
train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=False)

test_dataset = Samsung_Dataset_test("/data2/jaep0805/datasets/samsungdataset/CVPR2020_NTIRE_Workshop/test")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Build DeepHDR model from configs
model = DeepHDR(configs)
if configs.multigpu is False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), betas=(configs.beta1, configs.beta2), lr=configs.learning_rate)

# Define Criterion
criterion = HDRLoss()

# Define Scheduler
lr_scheduler = PolyLR(optimizer, max_iter=configs.epoch, power=0.9)

# Read checkpoints
start_epoch = 0
checkpoint_file = configs.checkpoint_dir + '/checkpoint.tar'
if configs.load == True:
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    print("Load checkpoint %s (epoch %d)", checkpoint_file, start_epoch)


if configs.multigpu is True:
    model = torch.nn.DataParallel(model)


def train_one_epoch(epoch):
    model.train()
    mean_train_loss = 0
    count = 0
    for idx, data in enumerate(train_dataloader):
        in_LDRs, in_HDRs, ref_HDRs = data
        image1 = in_LDRs[:, 0:3, :, :]
        image2 = in_LDRs[:, 3:6, :, :]
        image3 = in_LDRs[:, 6:9, :, :]
        image4 = in_HDRs[:, 0:3, :, :]
        image5 = in_HDRs[:, 3:6, :, :]
        image6 = in_HDRs[:, 6:9, :, :]

        in_LDRs = in_LDRs.to(device)
        in_HDRs = in_HDRs.to(device)
        ref_HDRs = ref_HDRs.to(device)
        
        # Forward
        result = model(in_LDRs, in_HDRs)
        # Backward
        loss = (criterion(tonemap(result), tonemap(ref_HDRs)))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_train_loss += loss.item()
        count += 1
        #print('--------------- Train Batch %d ---------------' % (idx + 1))
        #print('loss: %.12f' % loss.item())
    ref_HDRs = tonemap(ref_HDRs)
    res = tonemap(result)
    #tv.utils.save_image([image1[0],image2[0],image3[0], image4[0], image5[0], image6[0]], "/data2/jaep0805/DeepHDR-pytorch/trainimg/{:.3f}input.png".format(epoch), normalize = True, value_range = (-1,1))
    #tv.utils.save_image([ref_HDRs[0], res[0]], "/data2/jaep0805/DeepHDR-pytorch/trainimg/{:.3f}.png".format(epoch), normalize = True, value_range = (-1,1))
    mean_train_loss = mean_train_loss / count
    return mean_train_loss


def eval_one_epoch(epoch):
    model.eval()
    mean_loss = 0
    count = 0
    psnrlist = []
    for idx, data in enumerate(test_dataloader):
        in_LDRs, in_HDRs, ref_HDRs = data
        #sample_path = sample_path[0]
        if in_LDRs.shape[3] % 8 != 0:
            in_LDRs = in_LDRs[:, :, :, : (in_LDRs.shape[3] // 8 * 8)]
            in_HDRs = in_HDRs[:, :, :, : (in_HDRs.shape[3] // 8 * 8)]
            ref_HDRs = ref_HDRs[:, :, :, : (ref_HDRs.shape[3] // 8 * 8)]
        image1 = in_LDRs[:, 0:3, :, :]
        image2 = in_LDRs[:, 3:6, :, :]
        image3 = in_LDRs[:, 6:9, :, :]
        image4 = in_HDRs[:, 0:3, :, :]
        image5 = in_HDRs[:, 3:6, :, :]
        image6 = in_HDRs[:, 6:9, :, :]
        in_LDRs = in_LDRs.to(device)
        in_HDRs = in_HDRs.to(device)
        ref_HDRs = ref_HDRs.to(device)
        # Forward
        with torch.no_grad():
            res = model(in_LDRs, in_HDRs)
        # Compute loss
        with torch.no_grad():
            loss = criterion(tonemap(res), tonemap(ref_HDRs))
        #dump_sample(sample_path, res.cpu().detach().numpy())
        #print('--------------- Eval Batch %d ---------------' % (idx + 1))
        #print('loss: %.12f' % loss.item())
        mean_loss += loss.item()
        count += 1
        psnrlist.append(psnr((tonemap(ref_HDRs)[0].cpu().numpy() + 1.0) / 2.0, (tonemap(res)[0].cpu().numpy() + 1.0)/2.0))
    #normalize and tonemap
    print("avg psnr :", np.mean(psnrlist))
    ref_HDRs = tonemap(ref_HDRs)
    res = tonemap(res)
    if epoch % 10 == 0 :
        tv.utils.save_image([image1[0],image2[0],image3[0], image4[0], image5[0], image6[0]], "/data2/jaep0805/DeepHDR-pytorch/resultimg/{:.3f}input.png".format(epoch), normalize = True, value_range = (-1,1))
        tv.utils.save_image([ref_HDRs[0], res[0]], "/data2/jaep0805/DeepHDR-pytorch/resultimg/{:.3f}.png".format(epoch), normalize = True, value_range = (-1,1))
    
    mean_loss = mean_loss / count
    return mean_loss


def train(start_epoch):
    global cur_epoch
    for epoch in range(start_epoch, configs.epoch):
        cur_epoch = epoch
        print('**************** Epoch %d ****************' % (epoch + 1))
        print('learning rate: %f' % (lr_scheduler.get_last_lr()[0]))
        mean_train_loss = train_one_epoch(epoch)
        print('mean train loss: %.12f' % mean_train_loss)
        loss = eval_one_epoch(epoch)
        lr_scheduler.step()
        
        if configs.multigpu is False:
            save_dict = {'epoch': epoch + 1, 'loss': loss,
                         'optimizer_state_dict': optimizer.state_dict(),
                         'model_state_dict': model.state_dict(),
                         'scheduler': lr_scheduler.state_dict()
                         }
        else:
            save_dict = {'epoch': epoch + 1, 'loss': loss,
                         'optimizer_state_dict': optimizer.state_dict(),
                         'model_state_dict': model.module.state_dict(),
                         'scheduler': lr_scheduler.state_dict()
                         }
        if epoch % 200 == 0:
            torch.save(save_dict, os.path.join(configs.checkpoint_dir, 'checkpoint.tar'))
            torch.save(save_dict, os.path.join(configs.checkpoint_dir, 'checkpoint' + str(epoch) + '.tar'))
        print('mean eval loss: %.12f' % loss)


if __name__ == '__main__':
    train(start_epoch)
