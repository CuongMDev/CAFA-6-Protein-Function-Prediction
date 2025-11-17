import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        
        # LSTM branch
        self.embedding = nn.Embedding(20, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)

        # Linear branch
        self.linear = nn.Sequential(
            nn.Linear(150, 64),
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)  # multi-label or binary
        )
    
    def forward(self, seq_ids, features):
        # LSTM branch
        emb = self.embedding(seq_ids)
        _, (hn, _) = self.lstm(emb)
        lstm_out = hn[-1]               # shape: [batch, 128]
        
        # Linear branch
        feat_out = self.linear(features)

        # Combine
        combined = torch.cat([lstm_out, feat_out], dim=1)
        logits = self.classifier(combined)
        return logits
