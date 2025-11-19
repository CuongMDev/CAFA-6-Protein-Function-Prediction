# learning_rate_schedule.py
import torch

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Scheduler kết hợp warm-up và scheduler gốc.
    """
    def __init__(self, optimizer, warmup_epochs, base_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # warm-up: tăng dần từ 0 → lr_max
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # sau warm-up dùng scheduler gốc
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super().step(epoch)
        else:
            self.base_scheduler.step(epoch)


def get_scheduler(optimizer, scheduler_type="step", warmup_epochs=0, **kwargs):
    """
    Trả về learning rate scheduler với tùy chọn warm-up.

    Args:
        optimizer: torch.optim.Optimizer
        scheduler_type: "step", "cosine", "exponential", "reduce_on_plateau"
        warmup_epochs: số epoch warm-up
        kwargs: tham số cho scheduler
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "step":
        step_size = kwargs.get("step_size", 10)
        gamma = kwargs.get("gamma", 0.1)
        base_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "cosine":
        T_max = kwargs.get("T_max", 50)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    elif scheduler_type == "exponential":
        gamma = kwargs.get("gamma", 0.95)
        base_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_type == "reduce_on_plateau":
        mode = kwargs.get("mode", "min")
        factor = kwargs.get("factor", 0.1)
        patience = kwargs.get("patience", 5)
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=True
        )
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' not supported!")

    # Wrap với warm-up nếu cần
    if warmup_epochs > 0 and scheduler_type != "reduce_on_plateau":
        scheduler = WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
    else:
        scheduler = base_scheduler

    return scheduler


# ===========================
# Example usage
# ===========================
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim

    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step scheduler với 3 epoch warm-up
    scheduler = get_scheduler(optimizer, scheduler_type="step", step_size=5, gamma=0.5, warmup_epochs=3)

    for epoch in range(15):
        print(f"Epoch {epoch+1}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()
