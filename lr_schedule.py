# learning_rate_schedule.py
import math
import torch

class MyScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Warmup theo ratio + step-based decay (linear, cosine, exponential)
    Đảm bảo LR cuối cùng = base_lr * final_lr_ratio chính xác.
    """
    def __init__(self, optimizer, total_steps, warmup_ratio=0.1,
                 scheduler_type="cosine", gamma=0.1, final_lr_ratio=0.1, last_epoch=-1):

        self.scheduler_type = scheduler_type.lower()
        self.gamma = gamma
        self.warmup_steps = max(1, int(total_steps * warmup_ratio))
        self.total_steps = total_steps
        self.final_lr_ratio = final_lr_ratio

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []

        for base_lr in self.base_lrs:
            final_lr = base_lr * self.final_lr_ratio

            # ----------------
            # 1. Warmup phase
            # ----------------
            if step <= self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
                lrs.append(lr)
                continue

            # ----------------
            # 2. Decay phase
            # ----------------
            decay_step = step - self.warmup_steps
            decay_total = self.total_steps - self.warmup_steps
            t = decay_step / decay_total  # normalized: 0 → 1

            if self.scheduler_type == "linear":
                # linear decay từ base_lr → final_lr
                lr = base_lr + (final_lr - base_lr) * t

            elif self.scheduler_type == "cosine":
                # cosine, nhưng scale để cuối = final_lr
                cosine = 0.5 * (1 + math.cos(math.pi * t))
                # scale để cosine(0)=1, cosine(1)=final_lr/base_lr
                lr = final_lr + (base_lr - final_lr) * cosine

            elif self.scheduler_type == "exponential":
                # exponential thật sự luôn giảm đúng từ base_lr → final_lr
                # base_lr * k^(decay_total) = final_lr  => k = (final_lr/base_lr)^(1/decay_total)
                k = (final_lr / base_lr) ** (1 / decay_total)
                lr = base_lr * (k ** decay_step)

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
