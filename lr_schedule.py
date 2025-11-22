# learning_rate_schedule.py
import math
import torch

class MyScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Warmup theo ratio + step-based decay (linear, cosine, exponential)
    """
    def __init__(self, optimizer, total_steps, warmup_ratio=0.1,
                 scheduler_type="cosine", gamma=0.1, final_lr_ratio=0.0, last_epoch=-1):
        """
        Args:
            optimizer: optimizer
            total_steps: tổng step (batch*epoch)
            warmup_ratio: tỉ lệ warm-up (0~1)
            scheduler_type: "linear", "cosine", "exponential"
            gamma: dùng cho exponential decay
            final_lr_ratio: lr cuối / lr_max (linear/cosine)
        """
        self.scheduler_type = scheduler_type.lower()
        self.gamma = gamma
        self.warmup_steps = max(1, int(total_steps * warmup_ratio))
        self.final_lr_ratio = final_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                # linear warmup
                lr = base_lr * step / self.warmup_steps
            else:
                decay_step = step - self.warmup_steps
                decay_total = self.total_steps - self.warmup_steps
                if self.scheduler_type == "linear":
                    lr = base_lr - (base_lr - base_lr * self.final_lr_ratio) * decay_step / decay_total
                elif self.scheduler_type == "cosine":
                    lr = base_lr * 0.5 * (1 + math.cos(math.pi * decay_step / decay_total))
                    lr = max(lr, base_lr * self.final_lr_ratio)
                elif self.scheduler_type == "exponential":
                    lr = base_lr * (self.gamma ** decay_step)
                else:
                    raise ValueError(f"Scheduler type {self.scheduler_type} not supported")
            lrs.append(lr)
        return lrs

    def step(self, epoch=None):
        super().step(epoch)

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
    scheduler = MyScheduler(optimizer, scheduler_type="step",
                              total_steps=total_steps, warmup_ratio=0.1,
                              step_size=3, gamma=0.5)

    for step in range(total_steps):
        print(f"Step {step+1}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()
