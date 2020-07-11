import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np,os

class OrigamiDatasetGenerate(Dataset):

    def __init__(self, origami_dir, bg_dir,dataset_size=150 ,transform=None):

        self.bg_dir = bg_dir
        self.origami_dir = origami_dir
        self.dataset_size = dataset_size
        self.transform = transform

        self.orgami_list = [f for f in os.listdir(self.origami_dir) \
         if os.path.isfile(os.path.join(self.origami_dir, f))]
        
        self.bg_list = [f for f in os.listdir(self.bg_dir) \
         if os.path.isfile(os.path.join(self.bg_dir, f))]
        
    
    def merge(self,origami_loc, background_loc):
        origami = cv2.imread(origami_loc)
        origami_black_bg = origami.copy()
        background = cv2.imread(background_loc)
        background = cv2.resize(background, dsize=(origami.shape[0],origami.shape[1]), 
                                interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(origami, cv2.COLOR_BGR2HSV)
        for i in range(origami.shape[0]):
            for j in range(origami.shape[1]):
                if (50<hsv[i,j,0]<70 and 50<hsv[i,j,1]<=255 and 50<hsv[i,j,1]<=255) :
                    origami[i,j] = background[i,j]
                    origami_black_bg[i,j] = 0 

        origami = cv2.cvtColor(origami, cv2.COLOR_BGR2RGB)
        origami_black_bg = cv2.cvtColor(origami_black_bg, cv2.COLOR_BGR2RGB)
        return origami, origami_black_bg


    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):

        idx_bg = idx 
        idx_org = round((idx/len(self.bg_list))*len(self.orgami_list))    
        greenScreen_origami_path = os.path.join(self.origami_dir,
                                self.orgami_list[idx_org])

        bg_img_path = os.path.join(self.bg_dir,
                                self.bg_list[idx_bg])
        
        origami_merged, origami_black_bg = self.merge(greenScreen_origami_path,bg_img_path)
        sample = {'augmented': origami_merged, 'original': origami_black_bg}

        if self.transform:
            sample = self.transform(sample)

        return sample

class AddNoise(object):
    def __init__(self, mean, var, noise_typ = "gauss"):
        self.noise_typ = noise_typ
        self.mean = mean 
        self.var = var
    
    def __call__(self, sample):
        original, augmented = sample['original'], sample['augmented']

        if self.noise_typ == "gauss":
            row,col,ch= augmented.shape
            mean = self.mean
            var = self.var
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = noisy = np.clip(augmented + gauss, 0,1)

        if self.noise_typ == "s&p":
            row,col,ch = augmented.shape
            s_vs_p = self.mean
            amount = self.var
            out = np.copy(augmented)
            # Salt mode
            num_salt = np.ceil(amount * augmented.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in augmented.shape]
            out[coords] = 255

            # Pepper mode
            num_pepper = np.ceil(amount* augmented.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in augmented.shape]
            out[coords] = 0
            noisy = out
        
        return {'original': original,
                    'augmented': noisy}    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        original, augmented = sample['original'], sample['augmented']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        original = np.transpose(original, (2, 0, 1))
        augmented = np.transpose(augmented, (2, 0, 1))
        return {'original': torch.from_numpy(original).float(),
                'augmented': torch.from_numpy(augmented).float()}


class Grayscale(object):
    """Convert ndarrays in sample to 2D."""

    def __call__(self, sample):
        original, augmented = sample['original'], sample['augmented']
        original = torch.mean(original, dim=0)
        augmented = torch.mean(augmented, dim=0)
        return {'original': original, 'augmented': augmented}

class Resize(object):
    """ndarrays resize."""
    
    def __init__(self, size):
        self.size = size
        
    def __call__(self, sample):
        original, augmented = sample['original'], sample['augmented']
        original = cv2.resize(original, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        augmented = cv2.resize(augmented, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        return {'original': original, 'augmented':augmented }

class Normalize(object):
    
    def __call__(self, sample):
        original, augmented = sample['original'], sample['augmented']
        
        min_noisy,max_noisy = np.min(original),np.max(original)
        original = (original - min_noisy)/(max_noisy-min_noisy)
        
        min_noisy,max_noisy = np.min(augmented),np.max(augmented)
        augmented = (augmented - min_noisy)/(max_noisy-min_noisy)
        return {'original': original, 'augmented':augmented }

class RandomBackground(object):


    def __init__(self, bg_folder):
        self.root_dir = bg_folder
    
    def __call__(self, sample):
        original, augmented = np.rint(sample['original']*255), sample['augmented']
        original = original.astype(np.uint8)
        bg_image_filenames = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]

        i = np.random.randint(0,len(bg_image_filenames))
        bg = cv2.imread(os.path.join(self.root_dir,bg_image_filenames[i]))

        req_sh = original.shape
        bg = cv2.resize(bg, dsize=(req_sh[0],req_sh[1]))
        
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        # ret2,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ret2,mask = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
        
        new= cv2.bitwise_and(bg,bg, mask= mask)
        new= cv2.bitwise_or(original,new)
        
        new = cv2.cvtColor(new,cv2.COLOR_BGR2RGB)
        return {'original': original/255, 'augmented':new/255 }

class ValidationGenerate(Dataset):

    def __init__(self, root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_image_filenames = [f for f in os.listdir(self.root_dir) \
         if os.path.isfile(os.path.join(self.root_dir, f))]

    def __len__(self):
        return len(self.all_image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        inp_img_path = os.path.join(self.root_dir,
                                self.all_image_filenames[idx])
        sample = plt.imread(inp_img_path)       
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensorValidate(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = np.transpose(sample, (2, 0, 1))
        return torch.from_numpy(sample)

class ResizeValidate(object):
    """ndarrays resize."""
    
    def __init__(self, size):
        self.size = size
        
    def __call__(self, sample):
        sample = cv2.resize(sample, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        return sample

class NormalizeValidate(object):
    
    def __call__(self, sample):   
        
        min_noisy,max_noisy = np.min(sample),np.max(sample)
        sample = (sample - min_noisy)/(max_noisy-min_noisy)
        
        return sample
