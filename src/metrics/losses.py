import torch
import torch.nn.functional as F
from torch import nn


class Loss(nn.Module):
    def __init__(self, loss_fn: nn.Module) -> None:
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Loss module.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Computed loss value.
        """
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            # Convert y to one-hot encoding
            y = y.long()
            y_idx = F.one_hot(y, num_classes=x.shape[-1]).float()
            return self.loss_fn(x, y_idx)

        elif isinstance(self.loss_fn, nn.NLLLoss):
            _, _, vocab_size = x.shape
            return F.cross_entropy(
                x.reshape(-1, vocab_size),
                y.reshape(-1).long(),
                ignore_index=self.loss_fn.ignore_index,
            )

        else:
            raise ValueError("Unsupported loss function type")
