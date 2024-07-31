import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Define a simple CNN model with one convolutional layer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc = nn.Linear(1 * 14 * 14, 10, bias=False)  # Fully connected layer for classification
        self.fc = nn.Linear(1 * 28 * 28, 10, bias=False)  # Fully connected layer for classification

    def forward(self, x):
        x = self.conv1(x)  # 10000 * 1 * 28 * 28
        # x = self.relu(x)  # 10000 * 1 * 28 * 28
        # x = self.maxpool(x)  # 10000 * 1 * 14 * 14
        # x = x.view(-1, 1 * 14 * 14)  # Reshape to fit the fully connected layer # 10000 * (1*14*14)
        x = x.view(-1, 1 * 28 * 28)  # Reshape to fit the fully connected layer # 10000 * (1*14*14)
        x = self.fc(x)
        return x

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
    transforms.Normalize((0,), (255,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the CNN model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on test set: {accuracy:.2%}')

# save parameters
conv1_weight_ndarray = model.conv1.weight.data.numpy()
fc_weight_ndarray = model.fc.weight.data.numpy()

# conv1_weight_ndarray have shape 1 * 1* 3 * 3
conv1_weight_ndarray = np.reshape(conv1_weight_ndarray, [3, 3])
np.savetxt('conv1_weight.csv', conv1_weight_ndarray, delimiter=',')
np.savetxt('fc_weight.csv', fc_weight_ndarray, delimiter=',')
