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
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
                       
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of LSTMCell and fully connected layer.
        Biases are initialized to zero and weights using Xavier uniform initialization.
        """

        # TODO: Implement init_weights to initialize the weights of the LSTM and fully connected layer
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
        nn.init.constant_(self.fc.bias, 0.0)


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
        for t in range(T):
            x_t = x[:, t, :]
            h_t, c_t = self.lstm(x_t, (h_t, c_t))

        y = self.fc(h_t)
        return torch.sigmoid(y)
    

    def init_hidden(self, N):
        """
        Initializes the hidden state and cell state for LSTM with zeros.

        Args:
        N: Batch size

        Returns:
        A tuple of (hidden state, cell state), both initialized to zeros with shape (N, hidden_size).
        """
        # TODO: Return the initial hidden and cell states for LSTM
        device = next(self.parameters()).device
        h_0 = torch.zeros(N, self.hidden_size, device=device)
        c_0 = torch.zeros(N, self.hidden_size, device=device)
        return h_0, c_0
        
