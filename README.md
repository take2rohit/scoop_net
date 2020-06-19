# Pose Estimation of Origami using Denoising Autoencoders 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/take2rohit/denoising_autoencoder/blob/master/colab_denoising_ae.ipynb)

This code has been completely written from scratch using PyTorch library.
A sample of dataset has been already uploaded in folder of [sample_dataset](/home/rohit/projects/autoencoder/)

## How to use the repo:

1. Install `requirements.txt` by running

```bash 
pip3 install -r requirements.txt
```

2. Follow the Jupyter notebook **[denoising_ae.ipynb](denoising_ae.ipynb)** for complete code
3. For loading dataset of origami change the variable `origami_dataset_dir` to the directory containing folders of input and output.
4. Similarly change the variable `inp` and `out` to the directory containing various images containing Network's input and Output images repsectively.
```python
origami_dataset_dir = "MarowDataset"
inp='Input'
out='Output'
```
5. Dataset Directory info

```
    MarowDataset
    │
    ├── Input
    │   ├── img100_1.png
    │   ├── img100_2.png
    │   ├── ....
    |
    ├── Output
    │   ├── img100_1.png
    │   ├── img100_2.png
    │   ├──  ....
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
- Ruphan S
