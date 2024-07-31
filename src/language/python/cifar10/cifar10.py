import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from simpleNet import simpleNet
from resnet import ResNet9

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
# device = torch.device("cpu")

# 加载和预处理数据

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载训练和测试数据
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net = simpleNet().to(device)
net = ResNet9().to(device)
# 训练模型

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 记录训练时间
# start_train_time = time.time()
for epoch in range(100):  # 迭代次数
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): # total is 50000, batch is 100, i is 50000/100=500, idx of i is from 0 ~ 499.
        # 获取输入
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 将参数梯度清零
        optimizer.zero_grad()

        # 正向传播 + 反向传播 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        # print(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')

        running_loss += loss.item()
        # if i % 100 == 99:    # 每100个小批量打印一次
        #     print(f'[{epoch}, {i}] loss: {running_loss / 100:.3f}')
        #     running_loss = 0.0
    avg_loss = running_loss / 500
    print(f'[{epoch}] loss: {avg_loss:.3f}')
    with open('training_loss.txt', 'a') as f_loss:
        f_loss.write(f'{epoch} {avg_loss:.3f}\n')
    # 一轮epoch训练完了，评估模型
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'[{epoch}] Accuracy: {accuracy} %')
    with open('training_accuracy.txt', 'a') as f_accuracy:
        f_accuracy.write(f'{epoch} {accuracy:.2f} %\n')

# end_train_time = time.time()
# print(f'Finished Training in {end_train_time - start_train_time:.2f} seconds')

# 保存模型参数
torch.save(net.state_dict(), 'weight/model.pth')
print('Model parameters saved.')

# 加载模型参数
# net.load_state_dict(torch.load('./weight/model.pth'))
# net.eval()
# print('Model parameters loaded.')
#
# # 记录推理时间
# start_infer_time = time.time()
# #  评估模型
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# end_infer_time = time.time()
# print(f'Inference time: {end_infer_time - start_infer_time:.2f} seconds')
# print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
#

