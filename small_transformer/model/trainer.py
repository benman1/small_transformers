"""Training."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from small_transformer.base import ModelBase
from small_transformer.utils.data_loaders import PretokenizedDataLoader


class TransformerTrainer:
    """Trainer for training Transformer language models."""

    def __init__(
            self,
            model: ModelBase,
            train_data,
            val_data,
            lr: float = 1e-4,
            criterion = nn.CrossEntropyLoss(),
    ):
        """
        Args:
            model: Transformer model
            train_data: training text data
            val_data: validation text data
            lr: learning rate for Adam optimizer
            criterion: loss criterion
        """

        self.model = model
        self.train_loader = PretokenizedDataLoader(train_data, max_seq_len=128)
        self.val_loader = PretokenizedDataLoader(val_data, max_seq_len=128)
        self.criterion = criterion

        # Optimizer and scheduler
        self.opt = Adam(model.parameters(), lr)
        self.sch = CosineAnnealingLR(self.opt, 100000)

    def train(self, num_steps):
        """Training loop"""
        for step in range(num_steps):
            # Training step
            for x in self.train_loader:
                x = F.pad(x, (0, 128 - x.shape[1]))
                logits = self.model(x)
                loss = self.compute_loss(logits, x)

                loss.backward()
                self.clip_gradients()
                self.opt.step()
                self.opt.zero_grad()

            self.sch.step()

            # Evaluation
            val_loss = self.evaluate(self.val_loader)
            print(f"Step {step}: Train loss {loss:.4f}, Val loss {val_loss:.4f}")

    def compute_loss(self, logits, labels):
        """Compute cross entropy loss."""
        loss = self.criterion(logits, labels)
        return loss

    def evaluate(self, dataloader: PretokenizedDataLoader):
        """Evaluate average loss on data."""
        total_loss = 0.0
        for x in dataloader:
            logits = self.model(x)
            loss = self.compute_loss(logits, x)
            total_loss += loss
        return total_loss / len(dataloader)

    def clip_gradients(self, max_norm=1.0):
        """Gradient clipping to prevent exploding gradients."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
