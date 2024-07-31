import torch
import torch.nn as nn
import torch.quantization as quant
import torch.utils

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
        x = self.quant(x)
        x = self.conv1(x)
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc(x)
        x = self.dequant(x)
        return x

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





# Initialize your model
model = QuantizedCNN()

# Prepare for quantization
prepare_model_for_quantization(model)

# Create a calibration data loader (example using random data)
calibration_data = torch.randn(10000, 1, 28, 28)
calibration_labels = torch.randint(0, 10, (10000,))
calibration_dataset = torch.utils.data.TensorDataset(calibration_data, calibration_labels)
calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=32)

# Calibrate the model
calibrate_model(model, calibration_loader)

# Convert the model to quantized version
convert_model_to_quantized(model)



# print("quantisize module: ")
# print(model)

# The model is now quantized and ready for inference
# Example inference
example_input = torch.randn(1, 1, 28, 28)
model.eval()
with torch.no_grad():
    output = model(example_input)
print(output)


# unquantisation model
model_uq = CNN()

model_uq.eval()
with torch.no_grad():
    output = model_uq(example_input)
print(output)

