import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from datetime import date
from numpy.random import randint
from config2 import *
from sklearn.metrics import mean_absolute_error
from network import Net
from custom_dataset2 import CustomMNISTDataset
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch.backends import cudnn
from torchvision.transforms import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import *
from scipy.cluster.vq import vq, kmeans, whiten
from numpy import array

def main(type_experiment):
    test = datasets.MNIST(root='../data', train=False, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                          , download=True)

    test_dataset = CustomMNISTDataset(test.data, transform=transforms.ToTensor, apply_transform=False,
                                      experiment='mean')

    test_loader1 = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)
    test_loader2 = DataLoader(dataset=test, batch_size=BATCH_SIZE_TEST, shuffle=False)

    metric = np.array([[0, 0]])

    for idx, (_, label) in enumerate(test_loader1):
        for j in range(len(label)):
            metric = np.append(metric, [[0, label[j]]], axis=0)

    labels = np.array([[0]])

    for i in range(len(test.targets)):
        labels = np.append(labels, [[test.targets[i]]])

    for i in range(len(metric)):
        metric[i][0] = labels[i]

    # whitened = whiten(metric)

    codebook, distortion = kmeans(metric, 10)

    plt.scatter(metric[:, 0], metric[:, 1])
    plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    plt.show()



if __name__ == '__main__':
    main('')