import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, 
                 num_labels,
                 vocab_size=20,
                 embedding_dim=64,
                 transformer_hidden=128,
                 transformer_layers=2,
                 nhead=8,
                 linear_input_dim=150,
                 linear_hidden_dim=64,
                 classifier_hidden_dim=128):
        super().__init__()
        
        # --- Embedding for token sequences ---
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead, 
            dim_feedforward=transformer_hidden,
            batch_first=True,
            dropout=0.1,
            activation='relu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
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
            nn.Linear(embedding_dim + linear_hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, num_labels)
        )
        
    def forward(self, seq_ids, features, attn_mask=None):
        """
        seq_ids: [batch, seq_len]
        features: [batch, feature_dim]
        attn_mask: [batch, seq_len], 1=valid token, 0=padding
        """
        emb = self.embedding(seq_ids)  # [batch, seq_len, emb_dim]

        # --- Transformer requires src_key_padding_mask: True=pad, False=valid ---
        if attn_mask is not None:
            src_key_padding_mask = ~attn_mask.bool()  # invert mask
        else:
            src_key_padding_mask = None

        # --- Pass through Transformer ---
        transformer_out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)  # [batch, seq_len, emb_dim]

        # --- Pooling over tokens (mean of valid tokens) ---
        if attn_mask is not None:
            lengths = attn_mask.sum(dim=1).unsqueeze(1)  # [batch, 1]
            seq_out = (transformer_out * attn_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            seq_out = transformer_out.mean(dim=1)  # [batch, emb_dim]

        # --- Linear branch ---
        feat_out = self.linear(features)

        # --- Combine and classify ---
        combined = torch.cat([seq_out, feat_out], dim=1)
        logits = self.classifier(combined)

        return logits
