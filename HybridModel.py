import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, input_seq_dim, input_feat_dim, output_dim):
        super().__init__()
        
        # --- MLP cho embedding sequence (thay transformer) ---
        self.bn1 = nn.BatchNorm1d(input_seq_dim)
        self.fc1 = nn.Linear(input_seq_dim, 800)
        self.ln1 = nn.LayerNorm(800)
        self.activation = nn.Mish()
        
        self.bn2 = nn.BatchNorm1d(800)
        self.fc2 = nn.Linear(800, 600)
        self.ln2 = nn.LayerNorm(600)
        
        self.bn3 = nn.BatchNorm1d(600)
        self.fc3 = nn.Linear(600, 400)
        self.ln3 = nn.LayerNorm(400)
        
        # --- MLP cho features bổ sung ---
        self.bn_feat = nn.BatchNorm1d(input_feat_dim)
        self.fc_feat = nn.Linear(input_feat_dim, 400)
        self.ln_feat = nn.LayerNorm(400)
        
        # --- Classifier ---
        self.bn4 = nn.BatchNorm1d(800)  # 400 + 400
        self.fc4 = nn.Linear(800, output_dim)
        self.ln4 = nn.LayerNorm(output_dim)

    def forward(self, seq_emb, features):
        """
        seq_emb: [batch, input_seq_dim] -> embedding đã có sẵn
        features: [batch, input_feat_dim] -> features bổ sung
        """
        # --- Nhánh sequence ---
        x_seq = self.bn1(seq_emb)
        x_seq = self.activation(self.ln1(self.fc1(x_seq)))
        x_seq = self.bn2(x_seq)
        x_seq = self.activation(self.ln2(self.fc2(x_seq)))
        x_seq = self.bn3(x_seq)
        x_seq = self.activation(self.ln3(self.fc3(x_seq)))  # [batch, 400]
        
        # --- Nhánh features ---
        x_feat = self.bn_feat(features)
        x_feat = self.activation(self.ln_feat(self.fc_feat(x_feat)))  # [batch, 400]
        
        # --- Combine và classifier ---
        combined = torch.cat([x_seq, x_feat], dim=1)  # [batch, 800]
        x = self.bn4(combined)
        x = self.ln4(self.fc4(x))

        return x
