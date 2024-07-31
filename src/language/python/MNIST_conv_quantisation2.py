import torch
import torch.nn as nn
import torch.quantization as quant
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader, Subset


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = nn.Linear(1 * 28 * 28, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc(x)
        return x


class QuantizedCNN(CNN):
    def __init__(self):
        super(QuantizedCNN, self).__init__()
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        # print(x)
        x = self.quant(x)
        # print(x)
        x = self.conv1(x)
        # print(x)
        x = x.view(-1, 1 * 28 * 28)
        # print(x)
        x = self.fc(x)
        # print(x)
        x = self.dequant(x)
        # print(x)
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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (255,))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# Create a subset of the train_set for calibration (64 elements)
calibration_indices = list(range(64))  # Taking the first 64 elements
calibration_subset = Subset(train_set, calibration_indices)

# Create a DataLoader for the calibration dataset
calibration_loader = DataLoader(calibration_subset, batch_size=64, shuffle=False)


def prepare_model_for_quantization(model):
    model.qconfig = quant.get_default_qconfig('fbgemm')
    print(model.qconfig)
    quant.prepare(model, inplace=True)


def calibrate_model(model, calibration_loader):
    model.eval()
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model(inputs)


def convert_model_to_quantized(model):
    quant.convert(model, inplace=True)


def quantize_model(model):
    # Prepare for quantization
    prepare_model_for_quantization(model)
    # Calibrate the model
    calibrate_model(model, calibration_loader)
    # Convert the model to quantized version
    convert_model_to_quantized(model)


model_uq = CNN()
model_q = QuantizedCNN()

# Load the parameters from CSV files
load_parameters_from_csv('./conv1_weight.csv', 'fc_weight.csv', model_uq)
load_parameters_from_csv('./conv1_weight.csv', 'fc_weight.csv', model_q)

quantize_model(model_q)

start_time = time.time()
correct_q = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs_q = model_q(images)
        _, predicted_q = torch.max(outputs_q.data, 1)
        total += labels.size(0)
        correct_q += (predicted_q == labels).sum().item()
end_time = time.time()
time_q = end_time - start_time

accuracy_q = correct_q / total
print(f'Accuracy on test set in quantized model: {accuracy_q:.2%}, spend time: {time_q}')


start_time = time.time()
correct_uq = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs_uq = model_uq(images)
        _, predicted_uq = torch.max(outputs_uq.data, 1)
        total += labels.size(0)
        correct_uq += (predicted_uq == labels).sum().item()
end_time = time.time()
time_uq = end_time - start_time

accuracy_uq = correct_uq / total
print(f'Accuracy on test set in unquantized model: {accuracy_uq:.2%}, spend time: {time_uq}')
