"""
give some special case, to see if the scala and zero point is calculated by pytorch
in the way that I think.
"""
import torch
import torch.nn as nn
import torch.quantization as quant
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader, Subset


'''
input : 10 * 1 * 3 * 3
after conv1, the output is 10 * 1 * 2 * 2
after view, the output is 10 * 1 * 4
after fc, the output is 10 * 1
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(1 * 2 * 2, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 1 * 2 * 2)
        x = self.fc(x)
        return x


class QuantizedCNN(CNN):
    def __init__(self):
        super(QuantizedCNN, self).__init__()
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        print(x)
        x = self.quant(x)
        print(x)
        x = self.conv1(x)
        print(x)
        x = x.view(-1, 1 * 2 * 2)
        print(x)
        x = self.fc(x)
        print(x)
        x = self.dequant(x)
        print(x)
        return x


def load_parameters(conv1, fc, model):
    conv1_weight = conv1.view(model.conv1.weight.size())
    model.conv1.weight.data = conv1_weight

    fc_weight = fc.view(model.fc.weight.size())
    model.fc.weight.data = fc_weight

def prepare_model_for_quantization(model):
    model.qconfig = quant.get_default_qconfig('x86')
    # print(model.qconfig)
    quant.prepare(model, inplace=True)


def calibrate_model(model, calibration_loader):
    model.eval()
    with torch.no_grad():
        for inputs in calibration_loader:
            model(inputs[0])


def convert_model_to_quantized(model):
    quant.convert(model, inplace=True)


def quantize_model(model):
    # Prepare for quantization
    prepare_model_for_quantization(model)
    # Calibrate the model
    calibrate_model(model, calibration_loader)
    # Convert the model to quantized version
    convert_model_to_quantized(model)


def calcScaleZeroPoint(min_val, max_val, num_bits):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmin - min_val / scale

    zero_point = int(zero_point)

    return scale, zero_point


def quantize_tensor(x):
    min_val, max_val = min(x), max(x)
    num_bits = 8

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)

    q_x = [int(e / scale) + zero_point for e in x]
    return q_x


tensor_2x2 = torch.tensor([[-3., 2.], [32., 64.]])  # conv1 weight
tensor_4x1 = torch.tensor([[1.], [2.], [3.], [4.]])  # fc weight

input_tensor_3x3 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
input_tensor_3x3 = input_tensor_3x3.view(1, 1, 3, 3)
calibration_dataset = torch.utils.data.TensorDataset(input_tensor_3x3)
calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=1)

model_uq = CNN()
model_q = QuantizedCNN()

load_parameters(tensor_2x2, tensor_4x1, model_uq)
load_parameters(tensor_2x2, tensor_4x1, model_q)

quantize_model(model_q)

print("quantize inference")
model_q(input_tensor_3x3)
model_uq(input_tensor_3x3)

# print(model_q(input_tensor_3x3))
# print(model_uq(input_tensor_3x3))

print("end")