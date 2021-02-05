import torch as t
import torch.nn as nn
from torch.autograd import Variable
import cv2
from LeNet import LeNet5
import time
class model():
    def __init__(self, trainDataList, testDataList, lr = 0.001, loadModel = ''):
        self.trainDataList = trainDataList
        self.testDataList = testDataList
        self.lr = lr
        self.loadModel = loadModel
        self.net = LeNet5()
        # 加载模型
        if len(loadModel) != 0:
            self.net.load_state_dict(t.load(loadModel))
        self.use_gpu = t.cuda.is_available()
        if self.use_gpu:
            self.net=self.net.cuda()
            print("use GPU")
        else:
            print("use CPU")
        # 使用SGD优化
        self.optimizer=t.optim.SGD(self.net.parameters(), lr = 0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        # 记录训练损失值
        self.lossList = []
    def stratTrain(self, epoch):
        dataSize = 0
        for dataset in self.trainDataList:
            dataSize += len(dataset.dataset)
        for i in range(epoch):
            start=time.time()
            lossSum = 0
            for dataset in self.trainDataList:
                for data, label in dataset:
                    if self.use_gpu:
                        data,label=data.cuda(),label.cuda()
                    data,label=Variable(data),Variable(label)
                    self.optimizer.zero_grad()
                    output=self.net(data)
                    loss=self.criterion(output,label)
                    loss.backward()
                    self.optimizer.step()
                    lossSum += loss.item()
            lossSum /= dataSize
            now=time.time()
            print("第"+str(i+1)+"次迭代，loss="+str(lossSum)+ \
                            ",花费时间:"+str(now-start) + "s")
    def test(self):
        for i in range(len(self.testDataList)):
            correct = 0
            for data,label in self.testDataList[i]:
                if self.use_gpu:
                    data,label=data.cuda(),label.cuda()
                data,label=Variable(data),Variable(label)
                output=self.net(data)
                pred=output.data.argmax(1,keepdim=False)
                correct+=(pred==label).sum().item()
            print("在第" + str(i + 1) + "个测试集正确率为"+ \
                            str(100.*correct/len(self.testDataList[i].dataset)) + "%")

    def saveModel(self, savePath):
        t.save(self.net.state_dict(),savePath)

    def predict(self, imgPath):
        img = t.FloatTensor(cv2.resize(cv2.imread(imgPath, 0), (28, 28)).reshape(1, 1, 28, 28))/255
        if self.use_gpu:
            img = img.cuda()
        img = Variable(img)
        output = self.net(img)
        label = output.data.argmax(1,keepdim=False)
        return int(label[0])