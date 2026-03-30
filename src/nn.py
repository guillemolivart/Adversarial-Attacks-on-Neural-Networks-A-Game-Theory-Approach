import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicCNN(nn.Module):
    
    """
    A dynamically sized Convolutional Neural Network used as the Defender strategy.
    The architecture scales automatically based on the provided depth and width.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        input_size (int): Spatial dimensions of the input image (e.g., 32 for CIFAR-10).
        output_channels (int): Number of target classes.
        num_blocks (int): Depth of the network (number of Conv2d + ReLU + MaxPool blocks).
        base_channels (int): Width of the network (channels in the first convolutional layer).
        dropout_rate (float): Probability of an element to be zeroed in the fully connected layers.

    Returns:
        The output logits for each class (before softmax).
    """

    def __init__(
        self, 
        in_channels: int = 3, 
        input_size: int = 32, 
        output_channels: int = 10, 
        num_blocks: int = 2, 
        base_channels: int = 32, 
        dropout_rate: float = 0.5
    ) -> None:       
        super(DynamicCNN, self).__init__()
        
        self.features = nn.Sequential()
        
        current_channels = in_channels
        out_channels = base_channels
        current_size = input_size
        
        # Construïm la xarxa dinàmicament
        for i in range(num_blocks):
            self.features.add_module(f'conv_{i}', nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
            self.features.add_module(f'relu_{i}', nn.ReLU())
            self.features.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=2, stride=2))
            
            current_channels = out_channels
            out_channels *= 2  # Doblem els canals a cada bloc (ex: 32 -> 64 -> 128)
            current_size //= 2 # La mida es redueix a la meitat pel MaxPool

            assert current_size > 0, f"CRITICAL ERROR: So many ({num_blocks}) for an image of size {input_size}. The matrix has been reduced to 0."
            
        # Calculem automàticament la mida del tensor per a la capa Linear
        flatten_size = current_channels * current_size * current_size
        
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    

"""

class MLPModel(nn.Module):
    def __init__(self, in_channels = 3, input_size = 32, output_channels = 10, dropout_rate = 0.5):        
        super(MLPModel, self).__init__()

        self.input_channels = in_channels * input_size * input_size
        self.output_channels = output_channels
        
        self.dropout = nn.Dropout(p = dropout_rate)

        self.fc1 = nn.Linear(self.input_channels, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.output_channels)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # CrossEntropyLoss includes softmax, so we don't apply it here
        x = self.fc3(x) 
        return x
    
class CNNModel(nn.Module):
    def __init__(self, in_channels = 3, input_size = 32, output_channels = 10, dropout_rate = 0.5):
        super(CNNModel, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.output_channels = output_channels

        self.dropout = nn.Dropout(p = dropout_rate)

        self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels = 32, kernel_size = 5, padding = 1)

        size_conv = (self.input_size + 2*1 - 1*(5 - 1) - 1) // 1 + 1

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        size_pool = size_conv // 2

        self.fc1 = nn.Linear(in_features = 32 * size_pool * size_pool, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = self.output_channels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = torch.flatten(x, start_dim = 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # CrossEntropyLoss includes softmax, so we don't apply it here
        x = self.fc2(x)
        return x

"""