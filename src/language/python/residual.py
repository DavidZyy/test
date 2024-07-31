import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, main_path):
        super(Residual, self).__init__()
        self.main_path = main_path

    def forward(self, x):
        return x + self.main_path(x)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    main_path = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    res = Residual(main_path)
    result = nn.Sequential(res, nn.ReLU())
    return result

# Given input tensor
input_tensor = torch.tensor([[-0.41675785, -0.05626683, -2.1361961, 1.64027081, -1.79343559, -0.84174737,
                              0.50288142, -1.24528809, -1.05795222, -0.90900761, 0.55145404, 2.29220801,
                              0.04153939, -1.11792545, 0.53905832]])

# Define dimensions
dim = input_tensor.size(1)
hidden_dim = 10  # Example hidden dimension

# Instantiate the residual block
residual_block = ResidualBlock(dim, hidden_dim, nn.LayerNorm, 0.5)

# Get the output tensor
output_tensor = residual_block(input_tensor)

print(output_tensor)
