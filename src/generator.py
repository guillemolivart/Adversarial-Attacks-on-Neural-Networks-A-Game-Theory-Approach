import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGenerator(nn.Module):

    """
    An Autoencoder-based Adversarial Generator used as the Attacker strategy.
    It extracts features from an image and synthesizes targeted noise.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        input_size (int): Spatial dimensions of the input image.
        num_blocks (int): Depth of the encoder and decoder. Higher means larger receptive field.
        base_channels (int): Width of the network. Higher means more complex noise generation capacity.

    Returns:
        The adversarially perturbed image (x_adv) and the actual noise added (delta).
    """

    def __init__(
        self, 
        in_channels: int = 3, 
        input_size: int = 32, 
        num_blocks: int = 1, 
        base_channels: int = 16
    ) -> None:
        super(DynamicGenerator, self).__init__()
        
        self.in_channels = in_channels
        self.input_size = input_size
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        # Llista per saber quants canals tindrà cada bloc (ex: [16, 32, 64])
        enc_channels = [base_channels * (2**i) for i in range(num_blocks)]
        
        # ==========================================
        # CONSTRUCCIÓ DE L'ENCODER (Downsampling)
        # ==========================================
        current_in = in_channels
        for i in range(num_blocks):
            current_out = enc_channels[i]
            self.encoder.add_module(f'enc_conv_{i}', nn.Conv2d(current_in, current_out, kernel_size=3, stride=2, padding=1))
            self.encoder.add_module(f'enc_bn_{i}', nn.BatchNorm2d(current_out))
            self.encoder.add_module(f'enc_relu_{i}', nn.ReLU())
            current_in = current_out
            
        # ==========================================
        # CONSTRUCCIÓ DEL DECODER (Upsampling)
        # ==========================================
        # Donem la volta a la llista de canals per anar reconstruint (ex: [64, 32, 16])
        dec_channels = enc_channels[::-1] 
        
        for i in range(num_blocks):
            current_in = dec_channels[i]
            # L'últim bloc del decoder sempre ha de sortir amb 'base_channels'
            current_out = dec_channels[i+1] if i < num_blocks - 1 else base_channels
            
            # output_padding=1 és clau perquè el ConvTranspose2d dobli la mida exacta
            self.decoder.add_module(f'dec_convT_{i}', nn.ConvTranspose2d(current_in, current_out, kernel_size=3, stride=2, padding=1, output_padding=1))
            self.decoder.add_module(f'dec_bn_{i}', nn.BatchNorm2d(current_out))
            self.decoder.add_module(f'dec_relu_{i}', nn.ReLU())
            
        # ==========================================
        # CAPA FINAL (S'adapta a base_channels + in_channels de l'skip connection)
        # ==========================================
        self.final_layer = nn.Sequential(
            nn.Conv2d(base_channels + self.in_channels, self.in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, epsilon=0.1):
        # 1. Extraiem característiques (baixa resolució)
        encoded = self.encoder(x)
        
        # 2. Reconstruïm a la mida original
        decoded = self.decoder(encoded)
        
        # 3. Concatenem la imatge original amb les característiques processades
        combined = torch.cat((x, decoded), dim=1)
        
        # 4. Generem el soroll (-1 a 1 per la Tanh)
        noise = self.final_layer(combined)
        
        # 5. Escaleu el soroll i l'apliquem limitant valors (clamp)
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


class AdvancedPerturbationGenerator(nn.Module):
    
    #Encoder-Decoder architecture with upsampling layers.
    #Reduces the image to understand global features, then it reconstructs the image with the perturbation.
    
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

"""