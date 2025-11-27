from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm 
from config import DEVICE, GRADIENT_ACCUMULATION_STEPS, VAL_BATCH_SIZE, model_save_path, SCHEDULER_TYPE, GAMMA, WARMUP_RATIO, EPOCHS, \
                    log_step, val_step, learning_rate, weight_decay, \
                    BATCH_SIZE, VAL_RATIO, log_step, embedding_dim, top_k
from loss import MaskedWeightedBCE
from lr_schedule import MyScheduler
from load_data import load_train
import numpy as np
from custom_dataset import CustomDataset
from optimizer import SophiaG

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader=None,
                 criterion=None, optimizer=None, scheduler=None,
                 device='cuda', log_step=50, val_step=500,
                 model_save_path='best_model.pt', mask=None, top_k=None, weights=None):
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
        self.top_k = top_k
        self.weights = weights

        self.best_val_f1 = 0.0
        self.global_step = 0

        self.model.to(self.device)

        # Tổng số tham số
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def get_loss(self, batch, return_loss=True):
        seq_ids, features, mask, labels = batch
        seq_ids = seq_ids.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)

        outputs = self.model(seq_ids, features)
        
        if return_loss:
            loss = self.criterion(outputs, labels, mask=mask)
            return loss
        return outputs

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

        # Chuyển toàn bộ weights lên device, list: [BP_weights, MF_weights, CC_weights]
        all_weights = [torch.tensor(w, device=self.device) for w in self.weights]

        # Weighted F1 từng subontology
        weighted_f1_subont = []

        # Tính các index để slice probs/labels/weights theo top_k
        topk_cumsum = [0] + list(torch.cumsum(torch.tensor(self.top_k), dim=0).numpy())  # [0, k1, k1+k2, k_total]

        with torch.no_grad():
            # Tạo tqdm chung cho toàn bộ dataloader
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                logits = self.get_loss(batch, return_loss=False)  # [B, C_total]
                labels = batch[3].to(self.device)                 # [B, C_total]
                mask = batch[2].to(self.device)                   # [B, C_total]

                # Tính từng subontology
                for sub_idx, k in enumerate(self.top_k):
                    start, end = topk_cumsum[sub_idx], topk_cumsum[sub_idx+1]
                    weights_sub = all_weights[sub_idx]

                    probs = torch.sigmoid(logits[:, start:end])
                    labels_sub = labels[:, start:end]
                    mask_sub = mask[:, start:end]

                    # Áp dụng mask
                    probs_masked = probs * mask_sub
                    labels_masked = labels_sub * mask_sub

                    # Weighted counts
                    if len(weighted_f1_subont) <= sub_idx:
                        # Khởi tạo
                        weighted_f1_subont.append({
                            "tp": 0.0,
                            "pred": 0.0,
                            "actual": 0.0
                        })

                    weighted_f1_subont[sub_idx]["tp"] += (probs_masked * labels_masked * weights_sub).sum().item()
                    weighted_f1_subont[sub_idx]["pred"] += (probs_masked * weights_sub).sum().item()
                    weighted_f1_subont[sub_idx]["actual"] += (labels_masked * weights_sub).sum().item()

        # Sau khi duyệt hết batch, tính weighted F1 cho từng subontology
        final_f1 = []
        for sub_idx, counts in enumerate(weighted_f1_subont):
            tp, pred, actual = counts["tp"], counts["pred"], counts["actual"]
            precision = tp / (pred + 1e-12)
            recall = tp / (actual + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            final_f1.append(f1)

        # Trung bình 3 subontologies
        mean_weighted_f1 = sum(final_f1) / len(final_f1)

        # Lưu model nếu tốt nhất
        if mean_weighted_f1 > self.best_val_f1:
            self.best_val_f1 = mean_weighted_f1
            torch.save(self.model.state_dict(), self.model_save_path)
            tqdm.write(f"Step {self.global_step}: New best model saved with Mean Weighted F1: {mean_weighted_f1:.6f}")

        tqdm.write(f"Validation Mean Weighted F1: {mean_weighted_f1:.6f}")
        self.model.train()
        return mean_weighted_f1
        
def train(model, swa, train_dl, val_dl, evaluator, n_ep=20, lr=1e-3, clip_grad=1, weight_decay=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    scores = []
    for n in range(n_ep):

        model.train()
        for batch in tqdm.tqdm(train_dl):
            opt.zero_grad()

            output = model(batch)
            loss = loss_fn(output, batch['y'])
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad)
            opt.step()

        score = evaluator(model, val_dl)
        swa.add_checkpoint(model, score=score)
        print(f'Epoch {n}: CAFA5 score {score}')
        scores.append(score)

    return model, swa, scores
        
# --- Main ---
if __name__ == "__main__":
    from HybridModel import HybridModel
    import torch.optim as optim

    seq_data, feature_data, labels, mask, protein_vocab, term_vocab, ox_vocab, weights = load_train()
    print(f"num_labels: {len(term_vocab)}")
    print(f"vocab_size: {len(protein_vocab)}")
    print(f"linear_input_dim: {len(ox_vocab)}")
    
    dataset = CustomDataset(seq_data, feature_data, labels, mask, num_labels=len(term_vocab))

    # --- chia dataset ---
    train_size = int(len(dataset) * (1 - VAL_RATIO))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # --- tạo dataloader ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    model = HybridModel(
        output_dim=len(term_vocab), 
        input_seq_dim=embedding_dim, 
        input_feat_dim=len(ox_vocab)
    )
    criterion=MaskedWeightedBCE(weights=weights, reduction='mean', device=DEVICE)
    optimizer=SophiaG(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_updates = (len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS * EPOCHS  # ceil
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
        mask=mask,
        top_k=top_k,
        weights=weights
    )

    trained_model = trainer.train(num_epochs=EPOCHS, accumulate_steps=GRADIENT_ACCUMULATION_STEPS)