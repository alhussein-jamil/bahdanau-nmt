import torch
import torch.nn as nn
from torch.nn import init

from global_variables import DEVICE
from models.rnn import RNN


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        rnn_hidden_size = kwargs.get("rnn_hidden_size", 5)
        rnn_num_layers = kwargs.get("rnn_num_layers", 1)
        rnn_device = kwargs.get("rnn_device", "cpu")
        vocab_size = kwargs.get("vocab_size", 5)
        rnn_type = kwargs.get("rnn_type", "GRU")
        embedding_size = kwargs.get("embedding_size", 5)
        dropout = kwargs.get("dropout", 0.0)
        reverse_source = kwargs.get("reverse_source", True)
        super().__init__()
        self.vocab_size = vocab_size
        self.reverse_source = reverse_source
        self.rnn = RNN(
            input_size=embedding_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=rnn_device,
            dropout=dropout,
            bidirectional=True,
            type=rnn_type,
        )
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        init.normal_(self.embedding.weight, mean=0, std=0.01)

    @torch.autocast(DEVICE)
    def forward(self, x):
        # Paper Section 4.1: reverse source sentence for better memory in encoder RNN
        if self.reverse_source:
            x = torch.flip(x, dims=[1])
        embedded = self.embedding(x.long())
        with torch.autocast(DEVICE):
            rnn_output, rnn_hidden = self.rnn(embedded)
        return rnn_output, rnn_hidden
