import torch
from torch import nn
from torch.nn import init
from global_variables import DEVICE


class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) Module.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers.
        device (str): Device to which the model is moved (e.g., 'cuda' or 'cpu').
        activation (str, optional): Type of nonlinearity. Default is 'tanh'.
        dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer. Default is 0.

    Attributes:
        device (str): Device to which the model is moved.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        rnn (torch.nn.RNN): RNN layer.

    Methods:
        forward(x): Forward pass of the RNN.

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        device,
        dropout=0,
        bidirectional=False,
        type="RNN",
    ):
        """
        Initializes the RNN module.

        Parameters:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            device (str): Device to which the model is moved (e.g., 'cuda' or 'cpu').
            activation (str, optional): Type of nonlinearity. Default is 'tanh'.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer. Default is 0.
        """
        super(RNN, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        network_type = (
            nn.RNN if type == "RNN" else nn.LSTM if type == "LSTM" else nn.GRU
        )
        self.rnn = network_type(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # Initialize weights
        self.init_weights()

    @torch.autocast(DEVICE)
    def forward(self, x, h0=None):
        """
        Forward pass of the RNN.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Initialize hidden state
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers * (2 if self.rnn.bidirectional else 1),
                x.size(0),
                self.hidden_size,
                device=self.device,
                dtype=torch.float16,
            )
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(
                self.num_layers * (2 if self.rnn.bidirectional else 1),
                x.size(0),
                self.hidden_size,
                device=self.device,
                dtype=torch.float16,
            )
            out, hidden = self.rnn(x, (h0, c0))
        else:
            out, hidden = self.rnn(x, h0)
        return out.half(), (hidden[0] if isinstance(hidden, tuple) else hidden).half()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if "rnn" in name or "lstm" in name or "gru" in name or "hh" in name:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data, mean=0, std=0.01)
            if "bias" in name:
                init.constant_(param.data, val=0)
