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

    def __init__(self, root_dir, inp='Input', out='Output' ,transform=None):
        self.root_dir = root_dir
        
        self.inp_dir = inp
        self.inp_rel_path = os.path.join(root_dir,inp)
        
        self.out_dir = out
        self.out_rel_path = os.path.join(root_dir,out)

        self.transform = transform

        self.all_image_filenames = [f for f in os.listdir(self.inp_rel_path) \
         if os.path.isfile(os.path.join(self.inp_rel_path, f))]

    def __len__(self):
        return len(self.all_image_filenames)
    
    def image_augmentation(self, image, noise_typ = "s&p"):
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            
            return noisy

        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.1
            amount = 0.04
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
            out[coords] = 0
            return out    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inp_img_path = os.path.join(self.inp_rel_path,
                                self.all_image_filenames[idx])

        out_img_path = os.path.join(self.out_rel_path,
                                self.all_image_filenames[idx])
        
        in_img_path = plt.imread(inp_img_path)       
        out_img_path = plt.imread(out_img_path)
        
        sample = {'augmented': in_img_path, 'original': out_img_path}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
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