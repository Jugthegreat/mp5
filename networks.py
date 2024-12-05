import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, latent_size=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_size=32, hidden_size=128, output_size=2):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.decoder(x)




class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Downsampling path
        self.enc1 = self.conv_block(1, 64)   # Input channel is 1 for grayscale MNIST
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Upsampling path
        self.up3 = self.upconv_block(512, 256)
        self.up2 = self.upconv_block(256, 128)
        self.up1 = self.upconv_block(128, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # Output channel is 1 for grayscale MNIST

    def conv_block(self, in_channels, out_channels):
        """A convolutional block with Conv2D -> ReLU -> BatchNorm."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def upconv_block(self, in_channels, out_channels):
        """An upsampling block with ConvTranspose2D -> ReLU -> BatchNorm."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):
        device = inputs.device  # Get the device of the input
        batch = inputs.shape[0]
        inputs = inputs.reshape(batch, 1, 32, 32).to(device)  # Ensure inputs are on the same device
    
        # Ensure all layers are on the same device
        self.conv_block = self.conv_block.to(device)
        self.up1 = self.up1.to(device)
        self.up2 = self.up2.to(device)
        
        outputs = self.conv_block(inputs)  # Forward pass through UNet layers
        outputs = self.up1(outputs)
        outputs = self.up2(outputs)
        
        outputs = outputs.reshape(batch, -1)
        return outputs
