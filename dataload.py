from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch as t
class posionDataset(t.utils.data.Dataset):
    def __init__(self, imgPath, label, size, bias):
        np.random.seed(0)
        img = cv2.resize(cv2.imread(imgPath, 0), (28, 28)).reshape(1, 28, 28).astype(np.float32)
        self.data = []
        for i in range(size):
            self.data.append(t.FloatTensor(img + np.random.rand(1, 28, 28) * bias * 2 - bias) / 255)
        self.label = [label] * size
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)

def getStandardData(dataPath):
    train_dataset = datasets.MNIST(root=dataPath,
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    test_dataset = datasets.MNIST(root=dataPath,
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)
    train_data=DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_data=DataLoader(test_dataset,batch_size=128,shuffle=False)
    return train_data, test_data

def getInstanceKeyPoisonData(imgPath = 'x.jpg',imglabel = 8, trainSize = 5, testSize = 20, trainBias = 5, testBias = 5):
    trainSet = posionDataset(imgPath = 'x.jpg', \
        label = imglabel,size = trainSize, bias = 5)
    testSet = posionDataset(imgPath = 'x.jpg', \
        label = imglabel,size = testSize, bias = 20)
    trainData=DataLoader(trainSet,batch_size=64,shuffle=True)
    testData=DataLoader(testSet,batch_size=128,shuffle=False)
    return trainData, testData