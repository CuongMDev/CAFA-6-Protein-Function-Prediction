from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm 
from config import DEVICE, model_save_path, SCHEDULER_TYPE, GAMMA, WARMUP_RATIO, EPOCHS, \
                    lstm_hidden, lstm_layers, linear_hidden_dim, classifier_hidden_dim, learning_rate, weight_decay, \
                    BATCH_SIZE, VAL_RATIO, log_step, val_step
from loss import ASL
from lr_schedule import get_scheduler
from load_data import load_data
import numpy as np
from padding import CustomDataset, collate_fn

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader=None,
                 criterion=None, optimizer=None, scheduler=None,
                 device='cuda', log_step=50, val_step=500,
                 model_save_path='best_model.pt'):
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

        self.best_val_score = float('inf')
        self.global_step = 0

        self.model.to(self.device)

        # Tổng số tham số
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def get_loss(self, batch):
        seq_ids, features, labels = batch
        seq_ids = seq_ids.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(seq_ids, features)
        loss = self.criterion(outputs, labels)
        
        return loss

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            loop = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}",
                        unit="it", ncols=100)

            for batch_idx, batch in enumerate(self.train_dataloader):
                self.global_step += 1

                self.optimizer.zero_grad()
                loss = self.get_loss(batch)
                loss.backward()

                # --- grad norm ---
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += (p.grad.data.norm(2).item()) ** 2
                total_norm = total_norm ** 0.5

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                running_loss += loss.item() * batch[0].size(0)

                # --- logging ---
                if self.global_step % self.log_step == 0:
                    avg_loss = running_loss / ((batch_idx + 1) * self.train_dataloader.batch_size)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    tqdm.write(f"Step {self.global_step}\tTrain Loss: {avg_loss:.4f}\t"
                               f"Grad Norm: {total_norm:.4f}\tLR: {current_lr:.6f}")

                loop.update(1)

                # --- validation ---
                if self.val_dataloader is not None and self.global_step % self.val_step == 0:
                    self.validate_and_save()

            # --- cuối epoch ---
            epoch_train_loss = running_loss / len(self.train_dataloader.dataset)
            if self.val_dataloader is not None:
                epoch_val_loss = self.validate_and_save()
                print(f"Epoch {epoch+1}/{num_epochs} finished, "
                      f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} finished, Train Loss: {epoch_train_loss:.4f}")

            loop.close()

        return self.model

    def validate_and_save(self):
        self.model.eval()

        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                seq_ids, features, labels = batch
                seq_ids = seq_ids.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)

                # output shape: (batch_size, num_labels)
                logits = self.model(seq_ids, features)
                probs = torch.sigmoid(logits)

                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        # ghép tất cả batch lại
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        # --- Tính mAP ---
        # dùng macro để không bias label nhiều/no-label
        mAP = average_precision_score(all_labels, all_probs, average="macro")

        # --- Lưu model nếu tốt hơn ---
        if mAP > self.best_val_score:   
            self.best_val_score = mAP
            torch.save(self.model.state_dict(), self.model_save_path)
            tqdm.write(f"Step {self.global_step}: New best model saved with mAP {self.best_val_score:.4f}")

        self.model.train()
        return mAP
    
# --- Main ---
if __name__ == "__main__":
    from HybridModel import HybridModel
    import torch.optim as optim

    seq_data, feature_data, labels, protein_vocab, term_vocab, ox_vocab = load_data()
    print(f"num_labels: {len(term_vocab)}")
    print(f"vocab_size: {len(protein_vocab)}")
    print(f"linear_input_dim: {len(ox_vocab)}")
    
    dataset = CustomDataset(seq_data, feature_data, labels, num_labels=len(term_vocab))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # --- chia dataset ---
    train_size = int(len(dataset) * (1 - VAL_RATIO))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # --- tạo dataloader ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = HybridModel(
        num_labels=len(term_vocab), 
        vocab_size=len(protein_vocab), 
        linear_input_dim=len(ox_vocab), 
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        linear_hidden_dim=linear_hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim,
    )
    criterion=ASL()
    optimizer=optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- tính tổng step ---
    total_steps = len(dataloader) * EPOCHS  # số batch mỗi epoch × số epoch

    scheduler = get_scheduler(
        optimizer, 
        scheduler_type=SCHEDULER_TYPE, 
        total_steps=total_steps,      # truyền tổng step
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
        model_save_path=model_save_path
    )

    trained_model = trainer.train(num_epochs=EPOCHS)