import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, 
                 num_labels,
                 vocab_size=20,
                 embedding_dim=64,
                 lstm_hidden=128,
                 lstm_layers=1,
                 linear_input_dim=150,
                 linear_hidden_dim=64,
                 classifier_hidden_dim=128):
        super().__init__()
        
        # LSTM branch
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden, 
                            num_layers=lstm_layers, batch_first=True)

        # Linear branch
        self.linear = nn.Sequential(
            nn.Linear(linear_input_dim, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, 4 * linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * linear_hidden_dim, linear_hidden_dim)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + linear_hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, num_labels)  # multi-label hoặc binary
        )
    
    def forward(self, seq_ids, features):
        # LSTM branch
        emb = self.embedding(seq_ids)
        _, (hn, _) = self.lstm(emb)
        lstm_out = hn[-1]               # shape: [batch, lstm_hidden]
        
        # Linear branch
        feat_out = self.linear(features)

        # Combine
        combined = torch.cat([lstm_out, feat_out], dim=1)
        logits = self.classifier(combined)
        return logits
