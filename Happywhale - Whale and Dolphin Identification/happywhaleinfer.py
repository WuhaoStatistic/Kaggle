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
from HappyWhale_util import CONFIG, HappyWhaleDataset, HappyWhaleModel
# Sklearn Imports
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
# For Image Models
import timm
from sklearn.neighbors import NearestNeighbors

# # read the cvs file
df_test = pd.read_csv('test2.csv/test2.csv')
df_train = pd.read_csv('train2.csv/train2.csv')
# # read the model
# path = 'Loss10.8909_epoch6.bin'
# model = HappyWhaleModel(CONFIG['model_name'], CONFIG['embedding_size'])
# model.load_state_dict(torch.load(path))
# model.to('cuda')
# ## prepare dataloader
# # prepare transforms copied from training .py file
# data_transforms = {
#     "train": A.Compose([
#         A.Resize(CONFIG['img_size'], CONFIG['img_size']),
#         A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60),
#         A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
#         A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()], p=1.),
#
#     "valid": A.Compose([
#         A.Resize(CONFIG['img_size'], CONFIG['img_size']),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()], p=1.)
# }
#
# train_dataset = HappyWhaleDataset(df_train, transforms=data_transforms["valid"], trainflag=False)
# test_dataset = HappyWhaleDataset(df_test, transforms=data_transforms["valid"], trainflag=False)
# train_loader = DataLoader(train_dataset, batch_size=4, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=4, pin_memory=True)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# train_embeddings = []
# test_embeddings = []
# model.eval()
# with torch.no_grad():
#     for x in tqdm(train_loader):
#         images = x['image'].to(device, dtype=torch.float)
#         embedding = model.extract(images)
#         embedding = embedding.detach().cpu().numpy()
#         train_embeddings.append(embedding)
#     for x in tqdm(test_loader):
#         images = x['image'].to(device, dtype=torch.float)
#         embedding = model.extract(images)
#         embedding = embedding.detach().cpu().numpy()
#         test_embeddings.append(embedding)
#
# train_embeddings = np.concatenate(train_embeddings)
# test_embeddings = np.concatenate(test_embeddings)
#
# np.save('train_emb.npy', train_embeddings)
# np.save('test_emb.npy', test_embeddings)
# train_embeddings = np.load('train_emb.npy')
# test_embeddings = np.load('test_emb.npy')
# neigh = NearestNeighbors(n_neighbors=500, metric='cosine', n_jobs=-1)
# neigh.fit(train_embeddings)
# neigh.kneighbors()
#
# distances, idxs = neigh.kneighbors(test_embeddings, 500)

# np.save('distance.npy', distances)
# np.save('idxs.npy', idxs)

distances = np.load('distance.npy')
idxs = np.load('idxs.npy')

# distance is the distance,idxs is the corresponding index in the training_embeddings
df_res = pd.DataFrame(columns=['image', 'predictions'])
# df = df.append({'name':'yy', 'age':25, 'height':168},ignore_index=True)
# idx第i行就是测试集第i个样本的k近邻结果，每一行里的k个都是按顺序排号的训练集的index
# 排行第一的取过来，
for i in tqdm(range(idxs.shape[0])):
    img = df_test['image'][i]
    can = []
    remem = 0
    for x in range(500):
        # distances[i,x]就是第i个测试样例的knn结果里的第x个距离，对应的训练集样本index就是idxs[i,x]
        if distances[i, x] < 1e-4:
            if str(df_train['individual_id'][idxs[i, x]]) not in can:
                can.append(str(df_train['individual_id'][idxs[i, x]]))
        else:
            remem = x
            break
        if len(can) == 5:
            break
    # 插入 new_individual
    # if len(can) < 5:
    #     can.append('new_individual')
    already = len(can)
    while already < 5:
        if str(df_train['individual_id'][idxs[i, remem]]) not in can:
            can.append(str(df_train['individual_id'][idxs[i, remem]]))
            already += 1
        remem += 1
        if remem == idxs.shape[0]:
            break
        # 该样本结束
    res = ' '.join(can)
    df_res = df_res.append({'image': img, 'predictions': res}, ignore_index=True)
df_res.to_csv('submission_10.8909.csv', index=False)
