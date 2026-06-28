from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fcnn import FCNN
from models.rnn import RNN
from global_variables import maybe_autocast


class Alignment(nn.Module):
    def __init__(
        self,
        input_size: int,
        device: torch.device,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = input_size // 3

        self.nn_h = FCNN(
            input_size=self.hidden_size * 2,
            output_size=self.hidden_size,
            device=device,
            dropout=dropout,
            mean=0,
            std=0.001,
        )
        self.nn_s = FCNN(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            device=device,
            dropout=dropout,
            mean=0,
            std=0.001,
        )
        self.va = FCNN(
            input_size=self.hidden_size,
            output_size=1,
            device=device,
            dropout=dropout,
            mean=0,
            std=0.0,
        )

    def forward(self, s_emb: torch.Tensor, h_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Alignment module.

        Args:
            s_emb (torch.Tensor): Tensor representing the context from the decoder.
            h_emb (torch.Tensor): Tensor representing the hidden states from the encoder.

        Returns:
            torch.Tensor: Alignment vector.
        """
        with maybe_autocast():
            return self.va(F.tanh(s_emb + h_emb))


class OutputNetwork(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        max_out_units: int,
        hidden_size: int,
        vocab_size: int,
        device: torch.device,
        **kwargs,
    ) -> None:
        """
        Initializes the OutputNetwork module.

        Args:
            embedding_size (int): The size of the input embeddings.
            max_out_units (int): The maximum number of output units.
            hidden_size (int): The size of the hidden layers.
            vocab_size (int): The size of the vocabulary.
            device (torch.device): The device to be used for computation.
        """
        super().__init__()

        # matrix for s_i
        self.u_o = FCNN(
            input_size=hidden_size,
            output_size=2 * max_out_units,
            **kwargs,
        )

        # matrix for y_i
        self.v_o = FCNN(
            input_size=embedding_size,
            output_size=2 * max_out_units,
            **kwargs,
        )

        # matrix for c_i
        self.c_o = FCNN(
            input_size=2 * hidden_size,
            output_size=2 * max_out_units,
            **kwargs,
        )

        self.output_nn = FCNN(
            input_size=max_out_units,
            output_size=vocab_size,
            device=device,
            **kwargs,
        )
        self.output_size = vocab_size

    def forward(
        self, s_i: torch.Tensor, y_i: torch.Tensor, c_i: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a forward pass through the OutputNetwork module.

        Args:
            s_i (torch.Tensor): The input tensor s_i.
            y_i (torch.Tensor): The input tensor y_i.
            c_i (torch.Tensor): The input tensor c_i.

        Returns:
            torch.Tensor: The output tensor.
        """
        with maybe_autocast():
            # based on the article Maxout Networks
            t_tilde = (self.u_o(s_i) + self.v_o(y_i) + self.c_o(c_i))

            # sep odd and even
            t_even = t_tilde[:, 0::2]
            t_odd = t_tilde[:, 1::2]

            return self.output_nn(torch.max(t_even, t_odd))


class Decoder(nn.Module):
    def __init__(
        self,
        alignment: Dict[str, Any],
        rnn: Dict[str, Any],
        embedding: Dict[str, Any],
        output_nn: Dict[str, Any],
        traditional: bool = False,
        Ty: int = 10,
    ) -> None:
        super().__init__()
        # specify if we use the traditional encoder-decoder model
        self.traditional = traditional

        if traditional:
            rnn["input_size"] = rnn["hidden_size"] * 2

        # Alignment module
        self.alignment = Alignment(**alignment)

        self.hidden_size = rnn["hidden_size"]

        # GRU layer
        self.rnn = RNN(**rnn)

        self.vocab_size = embedding["vocab_size"]
        self.embedding = nn.Embedding(
            self.vocab_size, embedding["embedding_size"]
        )
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        # self.batch_norm_enc = nn.BatchNorm1d(2*rnn["hidden_size"])
        self.output_nn = OutputNetwork(**output_nn)

        # Used for the traditional model to return to the output size
        self.relaxation_nn = FCNN(
            input_size=rnn["hidden_size"],
            output_size=output_nn["vocab_size"],
        )
        self.Ty = Ty
        # Initialize context vector as a learnable parameter
        self.Ws = FCNN(
            input_size=rnn["hidden_size"],
            output_size=rnn["hidden_size"],
            device=embedding["device"],
        )


    def forward(self, t: int , h, h_emb = None,s_i = None, y_i = None):
        """
        Forward pass of the Decoder module.

        Args:
            t (int): The current time step.
            h (torch.Tensor): The hidden states from the encoder.
            h_emb (torch.Tensor): The embeddings of the hidden states from the encoder.
            s_i (torch.Tensor): The context from the decoder.
            y_i (torch.Tensor): The current output token.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Context vector.
            torch.Tensor: Alignment vector.
            
        """
        with maybe_autocast():
            if not self.traditional:
                if h_emb is None:
                    raise ValueError("h_emb must be specified for attention model")
                # Initialize context vector as a learnable parameter
                if s_i is None:
                    # Paper Eq. 5: s_0 = tanh(W_s * forward_h_1) — forward part at first position
                    s_i = F.tanh(self.Ws(h[:, 0, : self.rnn.hidden_size]))
                if y_i is None:
                    sos_token = self.vocab_size - 2
                    token_ids = torch.full(
                        (h.size(0),), sos_token, device=h.device, dtype=torch.long
                    )
                    embed_y_i = self.embedding(token_ids)
                elif y_i.is_floating_point() and y_i.shape[-1] == self.vocab_size:
                    embed_y_i = y_i.float() @ self.embedding.weight
                else:
                    token_ids = y_i.long() if y_i.dim() > 0 else y_i.long().unsqueeze(0)
                    embed_y_i = self.embedding(token_ids)
                
                # Compute the embedding of the current context vector
                s_i_emb = self.alignment.nn_s(s_i.view(h.size(0), -1))
                if self.rnn.device == "cuda":
                    s_i_emb = s_i_emb.half()
                
                # Compute alignment vector
                a = self.alignment(s_i_emb.unsqueeze(1).repeat(1, h.size(1), 1),
                                    h_emb).squeeze(2)
                

                # Apply softmax to obtain attention weights
                e = F.softmax(a.float(), dim=1)

                # Compute context vector
                c = torch.bmm(h.transpose(1, 2), e.unsqueeze(2)).squeeze(2)
                
                # Compute output and update context vector
                _, s_i = self.rnn(
                    torch.cat((embed_y_i.unsqueeze(1).float(),
                                c.unsqueeze(1).float()), dim=2),
                                  s_i.unsqueeze(0)
                )
                s_i = s_i.squeeze()

                # Embed the output token and compute the output of the output network
                y_i = self.output_nn(
                    s_i.view(h.size(0), -1), embed_y_i.squeeze(1), c
                )

               
                # Store the output in the output tensor
                return y_i, s_i, e
            y_i_emb, s_i = self.rnn(h[:, t,:].view(h.shape[0], 1, -1),s_i)
            y_i = self.relaxation_nn(y_i_emb)
            return y_i, s_i, None
