import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
import time

#  准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset01", train=True, transform=torchvision.transforms.ToTensor(),
                                       download=False)
test_data = torchvision.datasets.CIFAR10("./dataset01", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#  创建网络模型
model = Model01()

# 使用gpu
if torch.cuda.is_available():
    model = model.cuda()

#  损失函数
loss = nn.CrossEntropyLoss()
loss = loss.cuda()
# 优化器
learning_rate = 0.01
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

#  设置训练网络的一些参数
#  记录训练次数
total_train_step = 0
#  记录测试次数
total_test_step = 0
#  训练的次数
epoch = 50
#  添加tensorboard
writer = SummaryWriter("logs")
start_time = time.time()
for i in range(epoch):
    print("---————第 {} 轮训练开始----————".format(i+1))

    #  训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)
        result_loss = loss(outputs, targets)
        #  优化器优化模型
        optim.zero_grad()
        result_loss.backward()
        optim.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数：{}, Loss: {}".format(total_train_step, result_loss))
            writer.add_scalar("train_loss", result_loss, total_train_step)
    #  测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            result_loss = loss(outputs, targets)
            total_test_loss = total_test_loss + result_loss
            #  针对分类问题，可计算正确率
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每一轮的模型
    torch.save(model, "model_{}.pth".format(i))
    print("模型已保存")

writer.close()
