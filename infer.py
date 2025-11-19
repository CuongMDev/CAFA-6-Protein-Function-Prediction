import torch
from HybridModel import HybridModel
import torch.nn as nn

def load_model(model_path, num_labels, device):
    model = HybridModel(num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer(model, seq_ids, features, device):
    seq_ids = seq_ids.to(device)
    features = features.to(device)
    with torch.no_grad():
        outputs = model(seq_ids, features)
    
    
    outputs = torch.softmax(outputs, dim=-1)
    return outputs

if __name__ == "__main__":
    from config import DEVICE, model_save_path
    from torch.utils.data import DataLoader, TensorDataset

    # Dummy data for inference
    seq_data = torch.randint(0, 20, (10, 50))  # 10 sequences of length 50
    feature_data = torch.randn(10, 150)         # 10 samples with 150 features

    dataset = TensorDataset(seq_data, feature_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = load_model(model_save_path, num_labels=10, device=DEVICE)

    for seq_ids, features in dataloader:
        outputs = infer(model, seq_ids, features, device=DEVICE)
        print(outputs)