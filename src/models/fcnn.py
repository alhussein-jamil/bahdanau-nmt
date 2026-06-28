from typing import List

import torch
from torch import nn
from torch.nn import init

from global_variables import maybe_autocast


class FCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [],
        device: str = "cpu",
        activation: nn.Module = nn.Tanh(),
        last_layer_activation: nn.Module = nn.Identity(),
        dropout: float = 0,
        bias: bool = True,
        mean: float = 0,
        std: float = 0.01,
    ):
        super(FCNN, self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.last_layer_activation = last_layer_activation

        # Create a list of fully-connected layers
        self.fc = nn.ModuleList()

        # Add the first fully-connected layer
        self.fc.append(
            nn.Linear(
                self.input_size,
                self.hidden_sizes[0]
                if len(self.hidden_sizes) > 0
                else self.output_size,
                bias=bias,
            )
        )

        # Add the remaining fully-connected layers
        for i in range(1, len(self.hidden_sizes)):
            self.fc.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))

        if len(self.hidden_sizes) > 0:
            self.fc.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

        self.activation = activation

        # Add the dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize the weights
        self.init_weights(mean, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FCNN.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        with maybe_autocast():
            for i in range(len(self.fc) - 1):
                x = self.fc[i](x)
                x = self.activation(x)
                x = self.dropout(x)

            x = self.fc[-1](x)
            x = self.last_layer_activation(x)

        if self.device == "cuda":
            return x.half()
        return x

    def init_weights(self, mean: float = 0, std: float = 0.01):
        for name, param in self.named_parameters():
            if "weight" in name:
                if std == 0.0:
                    init.constant_(param.data, mean)
                else:
                    #init.xavier_normal_(param.data)
                    init.normal_(param.data, mean=mean, std=std)
            elif "bias" in name:
                init.constant_(param.data, 0)
