import torch
import torch.nn as nn
from config import DEVICE, model_save_path, SCHEDULER_TYPE, STEP_SIZE, GAMMA, WARMUP_EPOCHS
from lr_schedule import get_scheduler


def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for seq_ids, features, labels in dataloader:
            seq_ids = seq_ids.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(seq_ids, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * seq_ids.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    return model

if __name__ == "__main__":
    from HybridModel import HybridModel
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    # Dummy data
    seq_data = torch.randint(0, 20, (100, 50))  # 100 sequences of length 50
    feature_data = torch.randn(100, 150)         # 100 samples with 150 features
    labels = torch.randint(0, 2, (100, 10)).float()  # 100 samples, 10 labels (multi-label)

    dataset = TensorDataset(seq_data, feature_data, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = HybridModel(num_labels=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = get_scheduler(optimizer, scheduler_type=SCHEDULER_TYPE, step_size=STEP_SIZE, gamma=GAMMA, warmup_epochs=WARMUP_EPOCHS)


    trained_model = train_model(model, dataloader, criterion, optimizer,
                                 num_epochs=5, device=DEVICE)
    torch.save(trained_model.state_dict(), model_save_path)
