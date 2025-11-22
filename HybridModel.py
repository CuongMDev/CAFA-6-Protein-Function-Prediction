import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        
        # --- LSTM branch ---
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, lstm_hidden, 
            num_layers=lstm_layers, batch_first=True, bidirectional=False
        )

        # --- Linear branch ---
        self.linear = nn.Sequential(
            nn.Linear(linear_input_dim, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, 4*linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(4*linear_hidden_dim, linear_hidden_dim)
        )

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + linear_hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, num_labels)
        )
        
        # --- Attention ---
        self.attn_linear = nn.Linear(lstm_hidden, lstm_hidden)

    def forward(self, seq_ids, features, attn_mask=None):
        """
        seq_ids: [batch, seq_len]
        features: [batch, feature_dim]
        lengths: [batch] sequence lengths (without padding)
        attn_mask: [batch, seq_len], 1=valid token, 0=padding
        """
        emb = self.embedding(seq_ids)  # [batch, seq_len, emb_dim]
        
        # --- Pack sequences để LSTM ignore padding ---
        packed_emb = pack_padded_sequence(emb, attn_mask.sum(dim=-1).cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.lstm(packed_emb)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [batch, seq_len, hidden]
        
        # --- Attention token-to-token ---
        q = self.attn_linear(out)                    # [batch, seq_len, hidden]
        k = out.transpose(1, 2)                      # [batch, hidden, seq_len]
        attn_scores = torch.bmm(q, k) / (out.size(-1)**0.5)  # [batch, seq_len, seq_len]

        if attn_mask is not None:
            attn_mask = attn_mask.bool()  # [batch, seq_len]
            attn_mask_exp = attn_mask.unsqueeze(1).expand(-1, attn_mask.size(1), -1)  # [batch, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(~attn_mask_exp, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch, seq_len, seq_len]

        # --- Context vector per token ---
        context = torch.bmm(attn_weights, out)  # [batch, seq_len, hidden]

        # --- Mean pooling chỉ trên các token hợp lệ ---
        if attn_mask is not None:
            lengths = attn_mask.sum(dim=1).unsqueeze(1)  # [batch, 1]
            lstm_out = context.sum(dim=1) / lengths      # [batch, hidden]
        else:
            lstm_out = context.mean(dim=1)

        # --- Linear branch ---
        feat_out = self.linear(features)

        # --- Combine and classify ---
        combined = torch.cat([lstm_out, feat_out], dim=1)
        logits = self.classifier(combined)

        return logits
