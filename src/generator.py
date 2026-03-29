import torch
import torch.nn as nn
import torch.nn.functional as F

# Modify pixels individually, without understanding the global image structure.
class BasicPerturbationGenerator(nn.Module):
    """
    Maintains the same size as the input image (e.g., 32x32). 
    Output is the noise added to the original image.
    """
    def __init__(self, in_channels = 3, input_size = 32):
        super(BasicPerturbationGenerator, self).__init__()
        
        # Guardem les variables per consistència amb la resta de models
        self.in_channels = in_channels
        self.input_size = input_size

        self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = self.in_channels, kernel_size = 3, padding = 1)

    def forward(self, x, epsilon = 0.1):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        noise = torch.tanh(self.conv3(out))

        # Scale the noise because we don't want to add too much perturbation to the original image.
        delta = epsilon * noise

        x_adv = torch.clamp(x + delta, -1, 1)

        return x_adv, delta
    
class AdvancedPerturbationGenerator(nn.Module):
    """
    Encoder-Decoder architecture with upsampling layers.
    Reduces the image to understand global features, then it reconstructs the image with the perturbation.
    """
    def __init__(self, in_channels = 3, input_size = 32):
        super(AdvancedPerturbationGenerator, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # La capa final ara s'adapta dinàmicament als canals d'entrada
        self.final_layer = nn.Sequential(
            nn.Conv2d(16 + self.in_channels, self.in_channels, kernel_size = 3, padding = 1),
            nn.Tanh()
        )
    
    def forward(self, x, epsilon = 0.3):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        combined = torch.cat((x, decoded), dim = 1)
        noise = self.final_layer(combined)

        # Scale the noise because we don't want to add too much perturbation to the original image.
        delta = epsilon * noise

        x_adv = torch.clamp(x + delta, -1, 1)

        return x_adv, delta