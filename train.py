"""学習していないネットワークを使って予測する"""

import time
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import matplotlib.pyplot as plt
import models

ds_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.ToDtype(torch.float32, scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)

ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train, 
    batch_size=batch_size, 
    shuffle=True)

dataloader_test = torch.utils.data.DataLoader(
    ds_test, 
    batch_size=batch_size,
    shuffle=False 
    )



for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break

model = models.MyModel()

