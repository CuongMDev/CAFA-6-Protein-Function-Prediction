# learning_rate_schedule.py
import torch

class WarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Warm-up theo ratio kết hợp với scheduler gốc.
    """
    def __init__(self, optimizer, total_steps, warmup_ratio=0.1, base_scheduler=None, last_epoch=-1):
        """
        Args:
            optimizer: torch.optim.Optimizer
            total_steps: tổng số step (batches * epochs)
            warmup_ratio: tỉ lệ warm-up (0~1)
            base_scheduler: scheduler gốc (có thể None)
        """
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # linear warm-up: tăng từ 0 → lr_max
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            if self.base_scheduler is None:
                return [base_lr for base_lr in self.base_lrs]
            # base scheduler tính LR theo step
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        # tăng step
        super().step(epoch)
        # nếu base_scheduler có, advance base scheduler sau warm-up
        if self.base_scheduler is not None and self.last_epoch >= self.warmup_steps:
            self.base_scheduler.step(epoch)


def get_scheduler(optimizer, scheduler_type="step", total_steps=None, warmup_ratio=0.0, **kwargs):
    """
    Trả về scheduler với warm-up theo ratio.

    Args:
        optimizer: torch.optim.Optimizer
        scheduler_type: "step", "cosine", "exponential", "reduce_on_plateau"
        total_steps: tổng số step (batch * epochs), cần nếu warmup_ratio>0
        warmup_ratio: tỉ lệ warm-up (0~1)
        kwargs: tham số cho scheduler gốc
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

    # Wrap với warm-up ratio nếu cần
    if warmup_ratio > 0 and scheduler_type != "reduce_on_plateau":
        if total_steps is None:
            raise ValueError("total_steps must be provided when using warmup_ratio>0")
        scheduler = WarmupScheduler(optimizer, total_steps=total_steps,
                                    warmup_ratio=warmup_ratio, base_scheduler=base_scheduler)
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

    total_epochs = 10
    steps_per_epoch = 5
    total_steps = total_epochs * steps_per_epoch

    # Step scheduler với 10% warm-up
    scheduler = get_scheduler(optimizer, scheduler_type="step",
                              total_steps=total_steps, warmup_ratio=0.1,
                              step_size=3, gamma=0.5)

    for step in range(total_steps):
        print(f"Step {step+1}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()
