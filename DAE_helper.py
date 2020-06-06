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
import numpy as np,os

class OrigamiDatasetGenerate(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_image_filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.all_image_filenames)
    
    def image_augmentation(self, image, noise_typ = "gauss"):
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.05
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            
            min_noisy,max_noisy = np.min(noisy),np.max(noisy)
            noisy = (noisy - min_noisy)/(max_noisy-min_noisy)
            
            return noisy
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.all_image_filenames[idx])
        image = plt.imread(img_path)
        
        augmented = self.image_augmentation(image, noise_typ = "gauss")
        sample = {'augmented': augmented, 'original': image}

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
        original = original.transpose((2, 0, 1))
        augmented = augmented.transpose((2, 0, 1))
        return {'original': torch.from_numpy(original),
                'augmented': torch.from_numpy(augmented)}

class Grayscale(object):
    """Convert ndarrays in sample to 2D."""

    def __call__(self, sample):
        original, augmented = sample['original'], sample['augmented']
        
        original = torch.mean(original, dim=0)
#         original = torch.unsqueeze(original,0)
        
        augmented = torch.mean(augmented, dim=0)
#         augmented = torch.unsqueeze(augmented,0)
        
        return {'original': original, 'augmented': augmented}


