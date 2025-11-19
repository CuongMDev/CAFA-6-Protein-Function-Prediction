import torch
import torch.nn as nn

class ASL(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # clip negatives
        if self.clip:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # losses
        loss_pos = targets * torch.log(xs_pos) * (1 - xs_pos) ** self.gamma_pos
        loss_neg = (1 - targets) * torch.log(xs_neg) * xs_pos ** self.gamma_neg

        loss = -(loss_pos + loss_neg)
        return loss.mean()