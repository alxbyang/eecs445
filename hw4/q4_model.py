"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - HW4 - q4_models.py
Implement the Autoencoder class
"""
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple

class Autoencoder(nn.Module):
    def __init__(self, in_shape: Tuple[int, int], latent_dim: int=64):
        """
        Initialize the Autoencoder class

        Args:
        - in_shape: shape of the input image (height, width)
           | convert this to a 1-dimensional input size for the encoder!
        - latent_dim: size of the latent representation
        """
        super(Autoencoder, self).__init__()
        self.in_shape = in_shape

        # TODO: Implement the encoder and decoder as PyTorch modules 
        #       according to the specifications in problem set
        self.encoder = None
        self.decoder = None

        raise NotImplementedError

        self.apply(self._init_weights)

    def _init_weights(self, layer: nn.Module) -> None:
        """
        Initialize the weights of the layer

        Args:
        - layer: layer to initialize
        """
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input image into a latent representation of size latent_dim

        Args:
        - x: input image

        Returns: 
        - latent representation
        """
        # TODO: Implement the forward pass of the encoder
        raise NotImplementedError

    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation into the reconstructed image

        Args:
        - x: latent representation

        Returns:
        - reconstructed image
        """
        # TODO: Implement the forward pass of the decoder
        raise NotImplementedError


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder which encodes and decodes the input image

        Args:
        - x: input image

        Returns:
        - reconstructed image
        """
        # TODO: Implement the forward pass of the autoencoder
        raise NotImplementedError

        return x