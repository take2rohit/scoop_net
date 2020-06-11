import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class AugmentedAutoencoder(nn.Module):
    def __init__(self):
        super(AugmentedAutoencoder, self).__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.en_conv1 = nn.Conv2d(3, 512, 5, stride=2, padding=2) 
        self.en_conv2 = nn.Conv2d(512, 256, 5, stride=2, padding=2)
        self.en_conv3 = nn.Conv2d(256, 256, 5, stride=2, padding=2)
        self.en_conv4 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        
        self.fc1 = nn.Linear(512*8*8, 2048)
        self.fc2 = nn.Linear(2048, 512*8*8)
        
        self.dc_conv2 = torch.nn.ConvTranspose2d(512,256, 5, stride=2, padding=2, output_padding=1)
        self.dc_conv3 = torch.nn.ConvTranspose2d(256, 256, 5, stride=2, padding=2, output_padding=1)
        self.dc_conv4 = torch.nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1)
        self.dc_conv5 = torch.nn.ConvTranspose2d(128, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.en_conv1(x) # N x 128 x 64 x 64
        x = F.relu(x) 
        
        x = self.en_conv2(x) # N x 256 x 32 x 32
        x = F.relu(x)
        
        x = self.en_conv3(x) # N x 256 x 16 x 16
        x = F.relu(x)
        
        x = self.en_conv4(x) # N x 512 x 8 x 8
        x = F.relu(x)
        
        x = torch.flatten(x, 1) # N x (512*8*8)       
        encoding = self.fc1(x) # N x 128     
 
        x = self.fc2(encoding) # N x (512*8*8)
        x = x.view(-1,512,8,8)

        x = self.dc_conv2(x) # N x 256 x 16 x 16
        x = F.relu(x)

        x = self.dc_conv3(x) # N x 256 x 32 x 32
        x = F.relu(x)
        
        x = self.dc_conv4(x) # N x 256 x 64 x 64
        x = F.relu(x)

        x = self.dc_conv5(x) # N x 3 x 128 x 128
        x = F.sigmoid(x)
        return x
    
    def encoder_op(self,x):
        x = self.en_conv1(x) # N x 128 x 64 x 64
        x = F.relu(x) 
        
        x = self.en_conv2(x) # N x 256 x 32 x 32
        x = F.relu(x)
        
        x = self.en_conv3(x) # N x 256 x 16 x 16
        x = F.relu(x)
        
        x = self.en_conv4(x) # N x 512 x 8 x 8
        x = F.relu(x)
        
        
        x = torch.flatten(x, 1) # N x (512*8*8)       
        encoding = self.fc1(x) # N x 128
        return encoding

class DenoisingAE(nn.Module):
    def __init__(self):
        super(DenoisingAE,self).__init__()
        self.ec1 = nn.Linear(100*100, 512)
        self.ec2 = nn.Linear(512, 256)
        self.ec3 = nn.Linear(256, 128)
        self.ec4 = nn.Linear(128, 64)
        
        self.dc1 = nn.Linear(64, 128)
        self.dc2 = nn.Linear(128, 256)
        self.dc3 = nn.Linear(256, 512)
        self.dc4 = nn.Linear(512, 100*100)

    def forward(self, x):
        orig_size =  x.shape
        x = x.view(orig_size[0], -1).float()
        x = self.ec1(x)
        x = F.relu(x)
        x = self.ec2(x)
        x = F.relu(x)
        x = self.ec3(x)
        x = F.relu(x)
        x = self.ec4(x)
        encoded = F.relu(x)

        x = self.dc1(encoded)
        x = F.relu(x)
        x = self.dc2(x)
        x = F.relu(x)
        x = self.dc3(x)
        x = F.relu(x)
        x = self.dc4(x)
        reconst = F.sigmoid(x)
        return reconst

