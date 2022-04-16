import os
import gc

import copy
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Pytorch Imports
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler, optimizer
from torch.utils.data import Dataset, DataLoader

# Utils

from tqdm import tqdm
from collections import defaultdict
from HappyWhale_util import CONFIG, HappyWhaleDataset, HappyWhaleModel
# Sklearn Imports
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
# For Image Models
# Albumentations for augmentations
# For colored terminal text
from colorama import Fore, Back, Style
import warnings

b_ = Fore.BLUE
sr_ = Style.RESET_ALL
warnings.filterwarnings("ignore")


# For descriptive error messages
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


# print(df.kfold.value_counts())

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)

        batch_size = images.size(0)
        torch.cuda.empty_cache()

        outputs = model(images, labels)
        loss = criterion(outputs, labels)
        loss = loss / CONFIG['n_accumulate']
        loss.backward()

        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (float(loss) * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_loss


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images, labels)
        loss = criterion(outputs, labels)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    gc.collect()
    torch.cuda.empty_cache()
    return epoch_loss


def run_training(model, optimizer, scheduler, device, num_epochs, alreadyepo=0):
    # To automatically log gradients

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler,
                                           dataloader=tr_loader,
                                           device=CONFIG['device'], epoch=epoch)

        val_epoch_loss = valid_one_epoch(model, va_loader, device=CONFIG['device'],
                                         epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)

        # Log the metrics

        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "E:/pycharm/yolo5/Loss{:.4f}_epoch{:.0f}.bin".format(best_epoch_loss, epoch + alreadyepo)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def fetch_scheduler(optimizer, train_loader=None):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'],
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'],
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['learning_rate'],
                                            total_steps=CONFIG['epochs'] * len(train_loader))
    elif CONFIG['scheduler'] is None:
        return None

    return scheduler


def prepare_loaders(df, fold):
    df_train = df.reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = HappyWhaleDataset(df_train, transforms=data_transforms["train"])
    valid_dataset = HappyWhaleDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], pin_memory=True)

    return train_loader, valid_loader


if __name__ == '__main__':
    # freeze_support()
    torch.backends.cuda.cufft_plan_cache[0].max_size = 7000
    data_transforms = {
        "train": A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()], p=1.),

        "valid": A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()], p=1.)
    }
    set_seed(CONFIG['seed'])
    ROOT_DIR = 'E:/pycharm/yolo5/'
    TRAIN_DIR = ROOT_DIR + 'cropped_train_images/cropped_train_images/'
    TEST_DIR = ROOT_DIR + 'cropped_test_images/cropped_test_images/'
    df = pd.read_csv(ROOT_DIR + 'train2.csv/train2.csv')

    encoder = LabelEncoder()
    df['individual_id'] = encoder.fit_transform(df['individual_id'])
    skf = StratifiedKFold(n_splits=CONFIG['n_fold'])
    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.individual_id)):
        df.loc[val_, "kfold"] = fold

    # df = HappyWhaleDataset(df)
    model = HappyWhaleModel(CONFIG['model_name'], CONFIG['embedding_size'])
    # retrain the model is set True
    retrain = False
    if retrain:
        path = 'Loss11.0444_epoch5.bin'
        model.load_state_dict(torch.load(path))
        already = 5
        checkpoint = torch.load(path)
    model.to(CONFIG['device'])
    # do the k-fold
    for fo in range(CONFIG['n_fold']):
        torch.cuda.empty_cache()
        tr_loader, va_loader = prepare_loaders(df, fold=fo)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'],
                               weight_decay=CONFIG['weight_decay'])
        scheduler = fetch_scheduler(optimizer)
        model, history = run_training(model, optimizer, scheduler, CONFIG['device'], CONFIG['epochs'])
