import torch
import torch.nn as nn
import torch.nn.functional as F

# Modify pixels individually, without understanding the global image structure.
class BasicPerturbationGenerator(nn.Module):
    """
    Mantains the same size 28x28 as the input image. 
    Output is the noise added to the original image.
    """
    def __init__(self, in_channels = 1):
        super(BasicPerturbationGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = in_channels, kernel_size = 3, padding = 1)

    def forward(self, x, epsilon = 0.1):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        noise = torch.tanh(self.conv3(out))

        # Scale the noise because we don't want to add too much perturbation to the original image.
        delta = epsilon * noise

        x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv, delta
    
class AdvancedPerturbationGenerator(nn.Module):
    """
    Encoder-Decoder architecture with upsampling layers.
    Reduces the image to understand global features, then it reconstructs the image with the perturbation.
    """
    def __init__(self, in_channels = 1):
        super(AdvancedPerturbationGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, in_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Tanh()
        )
    
    def forward(self, x, epsilon = 0.1):
        encoded = self.encoder(x)
        noise = self.decoder(encoded)

        # Scale the noise because we don't want to add too much perturbation to the original image.
        delta = epsilon * noise

        x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv, delta