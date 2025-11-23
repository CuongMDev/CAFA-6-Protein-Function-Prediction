from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm 
from config import DEVICE, GRADIENT_ACCUMULATION_STEPS, VAL_BATCH_SIZE, model_save_path, SCHEDULER_TYPE, GAMMA, WARMUP_RATIO, EPOCHS, \
                    transformer_hidden, transformer_layers, nhead, linear_hidden_dim, classifier_hidden_dim, learning_rate, weight_decay, \
                    BATCH_SIZE, VAL_RATIO, log_step, val_step
from loss import ASL
from lr_schedule import MyScheduler
from load_data import load_train
import numpy as np
from custom_dataset import CustomDataset, collate_fn

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader=None,
                 criterion=None, optimizer=None, scheduler=None,
                 device='cuda', log_step=50, val_step=500,
                 model_save_path='best_model.pt', mask=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_step = log_step
        self.val_step = val_step
        self.model_save_path = model_save_path
        self.mask = mask

        self.best_val_loss = float('inf')
        self.global_step = 0

        self.model.to(self.device)

        # Tổng số tham số
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def get_loss(self, batch):
        seq_ids, features, mask, labels, attn_mask = batch
        seq_ids = seq_ids.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)
        attn_mask = attn_mask.to(self.device)

        outputs = self.model(seq_ids, features, attn_mask=attn_mask)
        loss = self.criterion(outputs, labels)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss

    def train(self, num_epochs=10, accumulate_steps=1):
        """
        accumulate_steps: số batch gom lại trước khi update optimizer
        """
        total_batches = len(self.train_dataloader)
        total_updates = (total_batches + accumulate_steps - 1) // accumulate_steps  # ceil
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            loop = tqdm(total=total_updates, desc=f"Epoch {epoch+1}/{num_epochs}",
                        unit="it", ncols=100)
            update_count = 0

            for batch_idx, batch in enumerate(self.train_dataloader):
                self.global_step += 1

                loss = self.get_loss(batch)
                loss = loss / accumulate_steps  # chia loss cho accumulate_steps
                loss.backward()

                # --- grad norm tính cho từng batch (tùy bạn muốn) ---
                if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == total_batches:
                    # --- grad norm ---
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += (p.grad.data.norm(2).item()) ** 2
                    total_norm = total_norm ** 0.5

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    running_loss += loss.item() * batch[0].size(0) * accumulate_steps  # nhân lại
                    update_count += 1
                    loop.update(1)  # update tqdm theo số update, không phải batch

                    # --- logging ---
                    if update_count % self.log_step == 0:
                        avg_loss = running_loss / ((batch_idx + 1) * self.train_dataloader.batch_size)
                        current_lr = self.optimizer.param_groups[0]['lr']
                        tqdm.write(f"Step {self.global_step}\tTrain Loss: {avg_loss:.7f}\t"
                                f"Grad Norm: {total_norm:.7f}\tLR: {current_lr:.6f}")

                    # --- validation ---
                    if self.val_dataloader is not None and update_count % self.val_step == 0:
                        self.validate_and_save()
                else:
                    running_loss += loss.item() * batch[0].size(0)

            # --- cuối epoch ---
            epoch_train_loss = running_loss / len(self.train_dataloader.dataset)
            if self.val_dataloader is not None:
                epoch_val_loss = self.validate_and_save()
                print(f"Epoch {epoch+1}/{num_epochs} finished, "
                    f"Train Loss: {epoch_train_loss:.7f}, Val Loss: {epoch_val_loss:.7f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} finished, Train Loss: {epoch_train_loss:.7f}")

            loop.close()

        return self.model

    def validate_and_save(self):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                loss = self.get_loss(batch)

                batch_size = batch[0].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        # loss mean trên dataset
        val_loss = total_loss / total_samples

        # ----- Save theo best loss -----
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.model_save_path)
            tqdm.write(f"Step {self.global_step}: New best model saved with val_loss {val_loss:.6f}")

        self.model.train()
        return val_loss
        
# --- Main ---
if __name__ == "__main__":
    from HybridModel import HybridModel
    import torch.optim as optim

    seq_data, feature_data, labels, mask, protein_vocab, term_vocab, ox_vocab = load_train()
    print(f"num_labels: {len(term_vocab)}")
    print(f"vocab_size: {len(protein_vocab)}")
    print(f"linear_input_dim: {len(ox_vocab)}")
    
    dataset = CustomDataset(seq_data, feature_data, labels, mask, num_labels=len(term_vocab))

    # --- chia dataset ---
    train_size = int(len(dataset) * (1 - VAL_RATIO))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # --- tạo dataloader ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = HybridModel(
        num_labels=len(term_vocab), 
        vocab_size=len(protein_vocab), 
        transformer_hidden=transformer_hidden,
        transformer_layers=transformer_layers,
        nhead=nhead,
        linear_input_dim=len(ox_vocab), 
        linear_hidden_dim=linear_hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim,
    )
    criterion=nn.BCEWithLogitsLoss(reduction='none')
    optimizer=optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_updates = (len(train_dataset) + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS * EPOCHS  # ceil
    scheduler = MyScheduler(
        optimizer, 
        total_steps=total_updates,
        scheduler_type=SCHEDULER_TYPE, 
        warmup_ratio=WARMUP_RATIO,             # ví dụ 10% tổng step là warm-up
        gamma=GAMMA
    )
    trainer = Trainer(
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,  # multi-label,   
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        log_step=log_step,
        val_step=val_step,
        model_save_path=model_save_path,
        mask=mask
    )

    trained_model = trainer.train(num_epochs=EPOCHS, accumulate_steps=GRADIENT_ACCUMULATION_STEPS)