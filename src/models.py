import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, input_size = 28, output_channels = 10):
        super(MLPModel, self).__init__()

        self.input_channels = input_size * input_size
        self.output_channels = output_channels

        self.fc1 = nn.Linear(self.input_channels, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.output_channels)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # CrossEntropyLoss includes softmax, so we don't apply it here
        x = self.fc3(x) 
        return x
    
class CNNModel(nn.Module):
    def __init__(self, in_channels = 1, input_size = 28, output_channels = 10):
        super(CNNModel, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.output_channels = output_channels

        self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels = 32, kernel_size = 5, padding = 1)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(in_features = 32 * 13 * 13, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = self.output_channels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = torch.flatten(x, start_dim = 1)

        x = F.relu(self.fc1(x))

        # CrossEntropyLoss includes softmax, so we don't apply it here
        x = self.fc2(x)
        return x
