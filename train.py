import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 
from config import DEVICE, model_save_path, SCHEDULER_TYPE, STEP_SIZE, GAMMA, WARMUP_RATIO, EPOCHS, \
                    lstm_hidden, lstm_layers, linear_hidden_dim, classifier_hidden_dim, learning_rate, weight_decay, \
                    BATCH_SIZE, VAL_RATIO, log_step, val_step
from lr_schedule import get_scheduler
from load_data import load_data
import numpy as np

# --- Custom Dataset để xử lý padding và one-hot ---
class CustomDataset(Dataset):
    def __init__(self, seq_data, feature_data, labels, num_labels):
        """
        seq_data: list of list of token ids
        feature_data: numpy array (b, linear_input_dim)
        labels: list of list các label
        num_labels: tổng số nhãn (size của one-hot)
        """
        self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in seq_data]
        self.feature_data = torch.tensor(feature_data, dtype=torch.float32)
        self.labels = [self.to_one_hot(lab, num_labels) for lab in labels]

    def to_one_hot(self, label_list, num_labels):
        one_hot = torch.zeros(num_labels, dtype=torch.float32)
        one_hot[label_list] = 1.0
        return one_hot

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        return self.seq_data[idx], self.feature_data[idx], self.labels[idx]

# --- collate_fn để pad trái sequence mỗi batch ---
def collate_fn(batch):
    seqs, features, labels = zip(*batch)
    # pad trái
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0, padding_side="left")  # default pad right, sẽ đổi thành pad trái
    max_len = seqs_padded.size(1)
    seqs_padded = torch.stack([torch.cat([torch.zeros(max_len - len(s), dtype=torch.long), s]) if len(s) < max_len else s for s in seqs])
    features = torch.stack(features)
    labels = torch.stack(labels)
    return seqs_padded, features, labels

# --- Training function ---
def train_model(model, train_dataloader, val_dataloader=None, criterion=None,
                optimizer=None, scheduler=None, num_epochs=10, device='cuda',
                log_step=50, val_step=500, model_save_path='best_model.pt'):

    model.to(device)
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", ncols=100)

        for batch_idx, (seq_ids, features, labels) in enumerate(train_dataloader):
            global_step += 1

            seq_ids = seq_ids.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(seq_ids, features)
            loss = criterion(outputs, labels)
            loss.backward()

            # tính grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * seq_ids.size(0)

            # --- cập nhật tqdm và log chỉ mỗi log_step ---
            if global_step % log_step == 0:
                avg_loss = running_loss / ((batch_idx + 1) * train_dataloader.batch_size)
                loop.set_postfix(loss=avg_loss, grad_norm=total_norm, lr=optimizer.param_groups[0]['lr'])
                loop.update(log_step)

            # --- validation theo val_step ---
            if val_dataloader is not None and global_step % val_step == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for seq_ids_val, features_val, labels_val in val_dataloader:
                        seq_ids_val = seq_ids_val.to(device)
                        features_val = features_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs_val = model(seq_ids_val, features_val)
                        loss_val = criterion(outputs_val, labels_val)
                        val_loss += loss_val.item() * seq_ids_val.size(0)

                epoch_val_loss = val_loss / len(val_dataloader.dataset)
                print(f"\nStep {global_step}: Val Loss: {epoch_val_loss:.4f}")

                # lưu model nếu tốt hơn
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Step {global_step}: New best model saved with val loss {best_val_loss:.4f}")

                model.train()  # trở lại train mode

        # --- log cuối epoch ---
        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        if val_dataloader is not None:
            # tính val loss cuối epoch
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq_ids_val, features_val, labels_val in val_dataloader:
                    seq_ids_val = seq_ids_val.to(device)
                    features_val = features_val.to(device)
                    labels_val = labels_val.to(device)

                    outputs_val = model(seq_ids_val, features_val)
                    loss_val = criterion(outputs_val, labels_val)
                    val_loss += loss_val.item() * seq_ids_val.size(0)
            epoch_val_loss = val_loss / len(val_dataloader.dataset)

            print(f"Epoch {epoch+1}/{num_epochs} finished, "
                  f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"Epoch {epoch+1}: New best model saved with val loss {best_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} finished, Train Loss: {epoch_train_loss:.4f}")

        loop.close()

    return model

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
    # Tổng số tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.BCEWithLogitsLoss()  # multi-label
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- tính tổng step ---
    total_steps = len(dataloader) * EPOCHS  # số batch mỗi epoch × số epoch

    # scheduler với warm-up ratio
    scheduler = get_scheduler(
        optimizer, 
        scheduler_type=SCHEDULER_TYPE, 
        total_steps=total_steps,      # truyền tổng step
        warmup_ratio=WARMUP_RATIO,             # ví dụ 10% tổng step là warm-up
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    trained_model = train_model(
        model, 
        train_loader, 
        val_loader,
        criterion, 
        optimizer, 
        scheduler,
        num_epochs=EPOCHS, 
        device=DEVICE, 
        log_step=log_step,
        val_step=val_step,
        model_save_path=model_save_path
    )