import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

class FaceLoader(Dataset):
    
    def __init__(self, csv_file, transform=None):
        """
        Custom dataset example for reading image locations and labels from csv
        but reading images from files
        
        Args:
            csv_path (string): path to csv file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Read the csv file
        df = pd.read_csv(csv_file)
        self.mlb = MultiLabelBinarizer()
        self.transform = transform 
        
        self.X_train = df['path']
        self.y_train = self.mlb.fit_transform(df['label'].str.split()).astype(np.float32)


    def __len__(self):
       return len(self.X_train.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name from the pandas df
        # Open image
        image = Image.open(self.X_train[idx])
        image = image.resize((256,256))
        #image = image - np.array([104,117,123])
        #image = image.transpose(2,0,1)

        pil2tensor = transforms.ToTensor()
        image = pil2tensor(image)
        
        label = torch.from_numpy(self.y_train[idx])
        sample = {'image': image, 'labels': label}

        if self.transform:
            sample = self.transform(image)

        return image, label
        
        

#dataset = FaceLoader(csv_file='./df.csv')


