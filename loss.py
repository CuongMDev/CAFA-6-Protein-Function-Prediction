import torch
import torch.nn as nn

class ASL(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, mask=None):
        """
        logits: (B, num_labels)
        targets: (B, num_labels), 0/1
        mask: (B, num_labels), bool or float
        """
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Clip negatives
        if self.clip:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Positive / negative losses
        loss_pos = targets * ((1 - xs_pos) ** self.gamma_pos) * torch.log(xs_pos + self.eps)
        loss_neg = (1 - targets) * (xs_pos ** self.gamma_neg) * torch.log(xs_neg + self.eps)

        # Total loss
        loss = -(loss_pos + loss_neg)  # negative log-likelihood

        # Apply mask
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum()  # mean over valid elements
        else:
            return loss.mean()