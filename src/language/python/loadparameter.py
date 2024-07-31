import torch
import torch.nn as nn
import pandas as pd
import torchvision
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = nn.Linear(1 * 28 * 28, 10, bias=False)  # Fully connected layer for classification

    def forward(self, x):
        x = self.conv1(x)  # 10000 * 1 * 28 * 28
        x = x.view(-1, 1 * 28 * 28)  # Reshape to fit the fully connected layer
        x = self.fc(x)
        return x

def load_parameters_from_csv(conv1_path, fc_path, model):
    # Load conv1 parameters
    conv1_weights = pd.read_csv(conv1_path, header=None).values
    conv1_weights = torch.tensor(conv1_weights, dtype=torch.float32)
    conv1_weights = conv1_weights.view(model.conv1.weight.size())
    model.conv1.weight.data = conv1_weights

    # Load fc parameters
    fc_weights = pd.read_csv(fc_path, header=None).values
    fc_weights = torch.tensor(fc_weights, dtype=torch.float32)
    fc_weights = fc_weights.view(model.fc.weight.size())
    model.fc.weight.data = fc_weights

# Instantiate the model
model = CNN()

# Load the parameters from CSV files
load_parameters_from_csv('./conv1_weight.csv', 'fc_weight.csv', model)

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
    transforms.Normalize((0,), (255,))
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

a = testset.data.numpy()
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
