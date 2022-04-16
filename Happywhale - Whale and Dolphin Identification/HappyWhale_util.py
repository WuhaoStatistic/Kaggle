import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from myutils import ArcMarginProduct
from myutils import GeM
import cv2
import torch.nn.functional as F
CONFIG = {"seed": 2021,
          "epochs": 6,
          'num_classes': 15587,
          "img_size": 512,
          "train_batch_size": 4,
          "valid_batch_size": 4,
          "model_name": "efficientnet_b4",
          "embedding_size": 512,
          "learning_rate": 8e-5,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-8,
          "T_max": 500,
          "weight_decay": 1e-7,
          "n_fold": 3,
          "margin": 0.2,
          "n_accumulate": 1,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  # output is cuda:0
          'number_neurons': 512,
          # ArcFace Hyperparameters
          "s": 30.0,
          "m": 0.30,
          "ls_eps": 0.0,
          "easy_margin": False
          }


class HappyWhaleDataset(Dataset):
    def __init__(self, df, transforms=None, trainflag=True):
        self.df = df
        self.file_names = df['imgpath'].values
        self.trainflag = trainflag
        if self.trainflag:
            self.labels = df['individual_id'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        if self.trainflag:
            label = self.labels[index]
            return {
                'image': img,
                'label': torch.tensor(label, dtype=torch.long)
            }
        return {
            'index': index,
            'image': img
        }


class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, embedding_size, pretrained=True):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.drop = nn.Dropout(0.2)
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ArcMarginProduct(embedding_size,
                                   CONFIG["num_classes"],
                                   s=CONFIG["s"],
                                   m=CONFIG["m"],
                                   easy_margin=CONFIG["ls_eps"],
                                   ls_eps=CONFIG["ls_eps"])

    def forward(self, images, labels):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.drop(pooled_features)
        embedding = self.embedding(embedding)
        output = self.fc(embedding, labels)
        return output

    def extract(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        embedding = F.normalize(embedding)
        return embedding

