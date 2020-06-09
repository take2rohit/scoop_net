

# Pose Estimation of Origami using Denoising Autoencoders 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VeJopZuGmnmk0Z2RQheKYcLWQjHL4cnq?usp=sharing)

This code has been completely written from scratch using PyTorch library.
A sample of dataset has been already uploaded in folder of [sample_dataset](/home/rohit/projects/autoencoder/)

## How to use the repo:

1. Install `requirements.txt` by running

```bash 
pip3 install -r requirements.txt
```

2. Follow the Jupyter notebook **[denoising_ae.ipynb](denoising_ae.ipynb)** for complete code
3. For loading dataset of origami change the variable `origami_dataset_dir` to the directory containing images of origami.
4. Similarly change the variable `random_background_dir` to the directory containing various background images. A random background will be picked everytime and will be applied to each image of dataset as dataset of image augmentation.

```python
origami_dataset_dir = "/home/rohit/Desktop/sample_test"
random_background_dir = '/home/rohit/Desktop/bg_samp'
```

5. There are various helper functions written by me to aid the code. These can be found in file `DAE_dataset_helper.py` and `DAE_model.py`

```python
## The classes imported below are used for dataloader, transformation and model

from DAE_dataset_helper import OrigamiDatasetGenerate
from DAE_dataset_helper import ToTensor,Resize, RandomBackground
from DAE_model import AugmentedAutoencoder
```

## Contributers

- Rohit Lal - [WEBSITE](http://take2rohit.github.io/)