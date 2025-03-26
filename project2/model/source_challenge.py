"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed


__all__ = ["Source"]


class Source(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: define each layer
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size= 5, stride = 2, padding = 2)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2, padding = 2)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 2, padding = 2)

        self.fc1 = nn.Linear(in_features = 128, out_features = 8)
        self.dropout = nn.Dropout(p=0.3)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        set_random_seed()

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            # TODO: initialize the parameters for the convolutional layers
            std = (1 / (5 * 5 * conv.in_channels)) ** 0.5
            nn.init.normal_(conv.weight, mean = 0.0, std = std)
            nn.init.constant_(conv.bias, 0.0)

        
        # TODO: initialize the parameters for [self.fc1]
        std = (1 / self.fc1.in_features) ** 0.5
        nn.init.normal_(self.fc1.weight, mean = 0.0, std = std)
        nn.init.constant_(self.fc1.bias, 0.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward propagation for a batch of input examples. Pass the input array
        through layers of the model and return the output after the final layer.

        Args:
            x: array of shape (N, C, H, W) 
                N = number of samples
                C = number of channels
                H = height
                W = width

        Returns:
            z: array of shape (1, # output classes)
        """
        N, C, H, W = x.shape

        # TODO: forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        
        return x