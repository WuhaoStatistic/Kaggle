import os
import gc
from multiprocessing.spawn import freeze_support

from efficientnet_pytorch import EfficientNet
import cv2
import copy
import time
import random
import matplotlib.pyplot as plt
# For data manipulation
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler, optimizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict
from HappyWhale_util import CONFIG
# Sklearn Imports
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
# For Image Models
import timm
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import PIL.Image as Image
import timm
from myutils import ArcMarginProduct
from myutils import GeM
import cv2

# Albumentations for augmentations
from torchvision import transforms as transforms
import PIL.Image as Image
from torchsummary import summary
# For colored terminal text
from colorama import Fore, Back, Style
import warnings
from species_based import Spe, Forspedataset, use_model

# read the cvs file
df_test = pd.read_csv('test2.csv/test2.csv')
# read the model
path = 'lose/Loss0.3042_rate9419.0000_epoch6.bin'
model = Spe(CONFIG['embedding_size'])
model.load_state_dict(torch.load(path))
model.to('cuda')
## prepare dataloader
# prepare transforms copied from training .py file
data_transforms = {
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()], p=1.)
}

test_dataset = Forspedataset(df_test, transforms=data_transforms["valid"], trainflag=False)
test_loader = DataLoader(test_dataset, batch_size=4, pin_memory=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
map1 = {0: 'beluga', 1: 'blue_whale', 2: 'bottlenose_dolphin', 3: 'brydes_whale', 4: 'commersons_dolphin',
       5: 'common_dolphin', 6: 'cuviers_beaked_whale', 7: 'dusky_dolphin', 8: 'false_killer_whale', 9: 'fin_whale',
       10: 'frasiers_dolphin', 11: 'globis', 12: 'gray_whale', 13: 'humpback_whale', 14: 'killer_whale',
       15: 'long_finned_pilot_whale', 16: 'melon_headed_whale', 17: 'minke_whale', 18: 'pantropic_spotted_dolphin',
       19: 'pilot_whale', 20: 'pygmy_killer_whale', 21: 'rough_toothed_dolphin', 22: 'sei_whale',
       23: 'short_finned_pilot_whale', 24: 'southern_right_whale', 25: 'spinner_dolphin', 26: 'spotted_dolphin',
       27: 'white_sided_dolphin'}

model.eval()
res = use_model(model, test_loader, CONFIG['device'], 0)
res = torch.tensor(res, device='cpu')
res = res.numpy().tolist()
df_test['species'] = res
df_test['species'] = df_test['species'].replace(map1)
df_test.to_csv('new_test.csv')