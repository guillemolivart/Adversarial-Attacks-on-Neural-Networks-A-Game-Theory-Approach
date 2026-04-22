import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DropBlock2d


class DynamicGenerator(nn.Module):

    """
    An Autoencoder-based Adversarial Generator used as the Attacker strategy.
    It extracts features from an image and synthesizes targeted noise.
    Supports dynamic structural scaling and ablation studies (regularization and normalization).
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        input_size (int): Spatial dimensions of the input image.
        num_blocks (int): Depth of the encoder and decoder. Higher means larger receptive field.
        base_channels (int): Width of the network. Higher means more complex noise generation capacity.
        reg_type (str): Type of spatial regularization to apply ('none', 'dropout', or 'dropblock').
        dropout_rate (float): Probability of an element to be zeroed. Set to 0.0 for no regularization.
        norm_type (str): Type of normalization layer to use ('none', 'batch', or 'instance').

    Returns:
        The adversarially perturbed image (x_adv) and the actual noise added (delta).
    """

    def __init__(
        self, 
        in_channels: int = 3, 
        input_size: int = 32, 
        num_blocks: int = 1, 
        base_channels: int = 16,
        reg_type: str = 'none',
        dropout_rate: float = 0.0,
        norm_type: str = 'none'
    ) -> None:
        super(DynamicGenerator, self).__init__()
        
        self.in_channels = in_channels
        self.input_size = input_size
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        # List to know how many channels each block will have (e.g., [16, 32, 64])
        enc_channels = [base_channels * (2**i) for i in range(num_blocks)]

        current_size = input_size
        
        # Encoder construction (Downsampling)
        current_in = in_channels
        for i in range(num_blocks):
            current_out = enc_channels[i]
            self.encoder.add_module(f'enc_conv_{i}', nn.Conv2d(current_in, current_out, kernel_size=3, stride=2, padding=1))
            
            current_size //= 2
            if norm_type == 'batch':
                self.encoder.add_module(f'enc_bn_{i}', nn.BatchNorm2d(current_out))
            elif norm_type == 'instance':
                self.encoder.add_module(f'enc_in_{i}', nn.InstanceNorm2d(current_out))
            
            self.encoder.add_module(f'enc_relu_{i}', nn.ReLU())

            if reg_type == 'dropout' and dropout_rate > 0:
                self.encoder.add_module(f'enc_drop2d_{i}', nn.Dropout2d(p=dropout_rate))
            elif reg_type == 'dropblock' and dropout_rate > 0:
                b_size = 3 if current_size >= 3 else 1
                self.encoder.add_module(f'enc_dropblock_{i}', DropBlock2d(p=dropout_rate, block_size=b_size))
            current_in = current_out
            
        # Decoder construction (Upsampling)
        dec_channels = enc_channels[::-1] 
        
        for i in range(num_blocks):
            current_in = dec_channels[i]
            # The last block of the decoder always has to output with 'base_channels'
            current_out = dec_channels[i+1] if i < num_blocks - 1 else base_channels
            
            # output_padding=1 is key because the ConvTranspose2d doubles the size exactly
            self.decoder.add_module(f'dec_convT_{i}', nn.ConvTranspose2d(current_in, current_out, kernel_size=3, stride=2, padding=1, output_padding=1))
            
            current_size *= 2

            if norm_type == 'batch':
                self.decoder.add_module(f'dec_bn_{i}', nn.BatchNorm2d(current_out))
            elif norm_type == 'instance':
                self.decoder.add_module(f'dec_in_{i}', nn.InstanceNorm2d(current_out))
            
            self.decoder.add_module(f'dec_relu_{i}', nn.ReLU())

            if reg_type == 'dropout' and dropout_rate > 0:
                self.decoder.add_module(f'dec_drop2d_{i}', nn.Dropout2d(p=dropout_rate))
            elif reg_type == 'dropblock' and dropout_rate > 0:
                b_size = 3 if current_size >= 3 else 1
                self.decoder.add_module(f'dec_dropblock_{i}', DropBlock2d(p=dropout_rate, block_size=b_size))
            
        # Final layer to generate noise, concatenating the original image with the decoder output
        self.final_layer = nn.Sequential(
            nn.Conv2d(base_channels + self.in_channels, self.in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, epsilon=0.1):
        # Extract features
        encoded = self.encoder(x)
        
        # Reconstruct to the original size
        decoded = self.decoder(encoded)
        
        # Concatenate the original image with the decoded features to give more context to the noise generation
        combined = torch.cat((x, decoded), dim=1)
        
        # Generate noise (-1 to 1 due to Tanh)
        noise = self.final_layer(combined)
        
        # Scale the noise and apply it
        delta = epsilon * noise
        x_adv = torch.clamp(x + delta, -1.0, 1.0)
        
        return x_adv, delta   

    

"""

# Modify pixels individually, without understanding the global image structure.
class BasicPerturbationGenerator(nn.Module):

    #Maintains the same size as the input image (e.g., 32x32). 
    #Output is the noise added to the original image.

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

"""

class AdvancedPerturbationGenerator(nn.Module):
    
    """
    An Encoder-Decoder architecture used to generate adversarial perturbations.
    It reduces the input image to understand global features, then reconstructs 
    it to generate the optimal noise (perturbation).
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images, 1 for grayscale).
        input_size (int): Spatial dimensions of the input image (e.g., 32 for CIFAR-10, 28 for MNIST).

    Returns:
        tuple: A tuple containing the adversarial image (x_adv) and the applied perturbation (delta).
    """

    def __init__(
        self, 
        in_channels: int = 3, 
        input_size: int = 32
    ) -> None:
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
    
    def forward(self, x: torch.Tensor, epsilon: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        combined = torch.cat((x, decoded), dim = 1)
        noise = self.final_layer(combined)

        # Scale the noise because we don't want to add too much perturbation to the original image.
        delta = epsilon * noise

        x_adv = torch.clamp(x + delta, -1, 1)

        return x_adv, delta
    
