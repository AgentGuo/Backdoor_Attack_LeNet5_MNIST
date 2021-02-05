# Backdoor Attacks Using Data Poisoning

[论文链接](https://arxiv.org/abs/1712.05526)

## 0. 模型说明

* 网络使用的是LeNet-5，只包含两个卷积层和若干全连接层，参数量很小
* 数据集使用的mnist手写数据集（训练集：60000  测试集：10000）

## 1. Input-instance-key strategies

直接使用一个输入实例作为后门，这里使用的是下图：

![x](./img/x.jpg)

然后生成投毒数据：

$$\sum_{rand}(x) = $$

## 2. Pattern-key strategies

使用一个识别模式作为一个后门

### 2.1 Blended Injection Strategy

### 2.2 Accessory Injection Strategy

### 2.3 Blended Accessory Injection Strategy

