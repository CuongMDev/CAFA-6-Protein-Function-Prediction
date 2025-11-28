import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, input_seq_dim, input_feat_dim, output_dim):
        super().__init__()

        self.activation = nn.Mish()

        self.features_emb = nn.Linear(input_feat_dim, 400)

        self.bn1 = nn.BatchNorm1d(input_seq_dim + 400)
        self.fc1 = nn.Linear(input_seq_dim + 400, 800)
        self.ln1 = nn.LayerNorm(800)

        self.bn2 = nn.BatchNorm1d(800)
        self.fc2 = nn.Linear(800, 600)
        self.ln2 = nn.LayerNorm(600)

        self.bn3 = nn.BatchNorm1d(600)
        self.fc3 = nn.Linear(600, 400)
        self.ln3 = nn.LayerNorm(400)

        self.bn4 = nn.BatchNorm1d(1200)

        self.out_ln1 = nn.Linear(1200, output_dim[0])
        self.out_ln2 = nn.Linear(1200, output_dim[1])
        self.out_ln3 = nn.Linear(1200, output_dim[2])
    def forward(self, inputs, extra_features=None):
        extra_features = self.features_emb(extra_features)

        inputs = torch.cat([inputs, extra_features], dim=-1)

        fc1_out = self.bn1(inputs)
        fc1_out = self.ln1(self.fc1(inputs))
        fc1_out = self.activation(fc1_out)

        x = self.bn2(fc1_out)
        x = self.ln2(self.fc2(x))
        x = self.activation(x)

        x = self.bn3(x)
        x = self.ln3(self.fc3(x))
        x = self.activation(x)

        x = torch.cat([x, fc1_out], dim=-1)
        x = self.bn4(x)
        
        x1 = self.out_ln1(x)
        x2 = self.out_ln2(x)
        x3 = self.out_ln3(x)

        x = torch.cat([x1, x2, x3], dim=-1)
        
        return x
