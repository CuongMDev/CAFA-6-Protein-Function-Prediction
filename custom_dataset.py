import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# --- Custom Dataset để xử lý padding và one-hot ---
class CustomDataset(Dataset):
    def __init__(self, seq_data, feature_data, labels=None, mask_fn=None, num_labels=None):
        """
        seq_data: np.array, mỗi phần tử là embedding vector (float)
        feature_data: numpy array (n_samples, feature_dim)
        labels: list của multi-label indices (list[int]) hoặc None
        mask_fn: hàm mask_fn(i) trả về list/array indices (lazy mask)
        num_labels: tổng số GO terms (kích thước one-hot)
        """
        # Chuyển seq_data sang tensor
        self.seq_data = torch.tensor(seq_data, dtype=torch.float32)
        
        self.feature_data = torch.tensor(feature_data, dtype=torch.float32)
        self.labels = labels
        self.mask_fn = mask_fn
        self.num_labels = num_labels

    def to_one_hot(self, indices):
        """Chuyển list indices thành one-hot tensor"""
        one_hot = torch.zeros(self.num_labels, dtype=torch.float32)
        if indices is not None and len(indices) > 0:
            one_hot[torch.tensor(indices, dtype=torch.long)] = 1.0
        return one_hot

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        seq = self.seq_data[idx]
        feat = self.feature_data[idx]

        # ----- MASK -----
        if self.mask_fn is not None:
            one_hot_mask = torch.from_numpy(self.mask_fn(idx)).float()
        else:
            one_hot_mask = None

        # ----- LABEL -----
        if self.labels is not None:
            indices_label = self.labels[idx]
            one_hot_label = self.to_one_hot(indices_label)
        else:
            one_hot_label = None

        # Return tuple (seq, feat, mask?, label?)
        if one_hot_mask is None and one_hot_label is None:
            return seq, feat
        if one_hot_mask is not None and one_hot_label is None:
            return seq, feat, one_hot_mask
        if one_hot_mask is None and one_hot_label is not None:
            return seq, feat, one_hot_label
        return seq, feat, one_hot_mask, one_hot_label