# Backdoor Attacks Using Data Poisoning

[论文链接](https://arxiv.org/abs/1712.05526)

## 0.GetStart

直接git clone整个仓库，新建一个`dataset`文件夹，Then run it！（其中的MNIST数据集会在线下载），直接运行`resultShow.ipynb`即可，如果需要训练运行`train.ipynb` :yum: 

文件目录如下

~~~
BACKDOOR_ATTACK_LENET5_MNIST
│  dataload.py
│  LeNet.py
│  LICENSE
│  model.py
│  README.md
│  resultShow.ipynb
│  train.ipynb
├─datasets
│  └─MNIST
│      ├─processed
│      │      test.pt
│      │      training.pt
│      │
│      └─raw
│              t10k-images-idx3-ubyte
│              t10k-images-idx3-ubyte.gz
│              t10k-labels-idx1-ubyte
│              t10k-labels-idx1-ubyte.gz
│              train-images-idx3-ubyte
│              train-images-idx3-ubyte.gz
│              train-labels-idx1-ubyte
│              train-labels-idx1-ubyte.gz
│
├─img
│      5.jpg
│      5_dot.jpg
│      dot.jpg
│      x.jpg
│
├─saveModel
       InstanceKeyLeNet.pth
       LeNet5.pth
       PatternKeyLeNet.pth
~~~

## 1. 模型说明

* 网络使用的是LeNet-5，只包含两个卷积层和若干全连接层，参数量很小
* 数据集使用的mnist手写数据集（训练集：60000  测试集：10000）
* 实现了Backdoor的两种攻击形式（instance-key和patten-key）

更多细节可以参考[博客]()