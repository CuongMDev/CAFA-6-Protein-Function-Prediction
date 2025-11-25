import torch
import torch.nn as nn

class MaskedWeightedBCE(nn.Module):
    """
    BCEWithLogitsLoss có hỗ trợ:
      - weight per term (IA weights)
      - mask per sample (ví dụ bỏ các label chưa biết)
      - tự chuẩn hoá weight khi khởi tạo
    """
    def __init__(self, weights=None, reduction='mean', device='cpu', log_normalize=True):
        """
        Args:
            weights: list hoặc tensor [n_terms], IA weights
            reduction: 'mean', 'sum' hoặc 'none'
            device: 'cpu' hoặc 'cuda'
            log_normalize: nếu True, log1p(weights) trước khi chuẩn hoá
        """
        super().__init__()
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32, device=device)
            if log_normalize:
                w = torch.log1p(w)  # log(1 + weight) giảm ảnh hưởng long tail
            # Chuẩn hoá theo mean để scale loss ~1
            w = w / (w.mean() + 1e-8)
            self.weights = w
        else:
            self.weights = None
            
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')  # no reduction here

    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: [batch_size, n_terms], output logits
            targets: [batch_size, n_terms], 0/1
            mask: [batch_size, n_terms], 0/1 -> 0: bỏ term này, 1: dùng term
        Returns:
            scalar loss
        """
        loss = self.criterion(logits, targets)  # [batch, n_terms]

        # Apply per-term weights
        if self.weights is not None:
            loss = loss * self.weights  # broadcast theo dim=0 của term

        # Apply mask
        if mask is not None:
            loss = loss * mask  # mask=0 → bỏ term
            # tránh chia cho 0
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # trả về raw per-term loss
