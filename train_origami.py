import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np,os
from DAE_dataset_helper import OrigamiDatasetGenerate,ValidationGenerate
from DAE_dataset_helper import ToTensor,Resize, Normalize
from DAE_dataset_helper import ToTensorValidate,NormalizeValidate,ResizeValidate
from DAE_model import AugmentedAutoencoder # contains various models to be tested on 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Trainer')

parser.add_argument("-e", "--epoch", type=int, help="epoch",default=100)
parser.add_argument("-tr", "--train",type=int, help="train batch size",default=1)
parser.add_argument("-te", "--test", type=int, help="test batch size",default=1)
parser.add_argument("-pre", '--pretrain', type=bool, help="use prev saved model",default=False)
args = vars(parser.parse_args())

epochs = args['epoch']
train_batch_size = args['train']
test_batch_size = args['test']
split_percent = 0.8
save_model = True
load_model = args['pretrain']
saved_pth = 'AE.pt'

origami_dataset_dir = "MarowDataset"
inp='Input'
out='Output'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

trns = transforms.Compose([Resize((128,128)), Normalize(),ToTensor()])
origami = OrigamiDatasetGenerate(root_dir=origami_dataset_dir,inp=inp, out=out, transform=trns)

train_size = int(split_percent * len(origami))
test_size = abs(len(origami) - train_size)
train_dataset, test_dataset = torch.utils.data.random_split(origami, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                            shuffle=True,**kwargs)

test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                            shuffle=True,**kwargs)

def train(model, device, train_loader, optimizer, epoch,log_interval=12):
    model.train()
    l2 = nn.MSELoss()
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['augmented'],sample['original']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = l2(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch,loss.item()))
    scheduler.step()

def test(model, device, test_loader,save_img=False,ep=None,save_folder=None):
    model.eval()
    l2 = nn.MSELoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for c, sample in enumerate(test_loader):
            data, target = sample['augmented'],sample['original']
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += l2(output,target) # sum up batch loss
    test_loss /= max(1,c)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    if save_img:
        kwargs =  {'nrow':4, "padding":2, "pad_value":1}
        images_show = 8
        op = output[:images_show,:].cpu()
        tar = data[:images_show,:].cpsu()
        cat = torch.cat((op,tar),dim=0)
        torchvision.utils.save_image(cat, os.path.join(save_folder, f'{ep}_in.png'),**kwargs)
        print('image saved')



model = AugmentedAutoencoder().to(device)

if os.path.exists(saved_pth) and load_model:
    model.load_state_dict(torch.load(saved_pth))

optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

if not os.path.exists('trained_results'):
    os.mkdir('trained_results')

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    
    if save_model and epoch%10 ==0:
        test(model, device, test_loader,save_img=True, ep = epoch, save_folder='trained_results')
        torch.save(model.state_dict(), saved_pth)