import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN model. Defines LSTMCell and fully connected layer.

        Args:
        input_size: Dimension of the input features.
        hidden_size: Dimension of the hidden state in the LSTM.
        output_size: Dimension of the output layer.
        """

        super().__init__()
        # TODO: Define the parameters of the RNN 
                            
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of LSTMCell and fully connected layer.
        Biases are initialized to zero and weights using Xavier uniform initialization.
        """

        # TODO: Implement init_weights to initialize the weights of the LSTM and fully connected layer

    def forward(self, x):
        """
        Forward pass of the model. Processes the input sequence through the LSTM
        and returns the final output after applying sigmoid.

        Args:
        x: Input tensor of shape (N, T, d), where N is batch size,
           T is sequence length, and d is input feature dimension.

        Returns:
        Output tensor after processing the sequence through LSTM and the fully connected layer.
        """
        N, T, d = x.shape

        # Initialize the hidden state and cell state
        h_t, c_t = self.init_hidden(x.size(0))

        # TODO: Loop through the sequence and apply LSTM to each time step
        

    def init_hidden(self, N):
        """
        Initializes the hidden state and cell state for LSTM with zeros.

        Args:
        N: Batch size

        Returns:
        A tuple of (hidden state, cell state), both initialized to zeros with shape (N, hidden_size).
        """
        # TODO: Return the initial hidden and cell states for LSTM
        
