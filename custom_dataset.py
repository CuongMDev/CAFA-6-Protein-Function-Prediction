import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# --- Custom Dataset để xử lý padding và one-hot ---
class CustomDataset(Dataset):
    def __init__(self, seq_data, feature_data, labels=None, mask_fn=None, num_labels=None):
        """
        seq_data: list of list token ids
        feature_data: numpy array (n_samples, feature_dim)
        labels: list of multi-label indices (list[int]) hoặc None
        mask_fn: hàm mask_fn(i) trả về list/array indices (lazy mask)
        num_labels: tổng số GO terms (kích thước one-hot)
        """
        self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in seq_data]
        self.feature_data = torch.tensor(feature_data, dtype=torch.float32)

        self.labels = labels
        self.mask_fn = mask_fn    # ⬅️ truyền hàm lazy mask
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
            one_hot_mask = torch.from_numpy(self.mask_fn(idx))  # kiểu list hoặc array
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
    
# --- collate_fn để pad trái sequence mỗi batch ---

def collate_fn(batch):
    """
    batch: list các tuple trả về từ CustomDataset.
    Có thể gồm:
      (seq, feat)
      (seq, feat, mask)
      (seq, feat, label)
      (seq, feat, mask, label)
    Trả thêm attention mask: 1 cho token thực, 0 cho padding
    """
    seqs = []
    feats = []
    masks = []
    labels = []

    for item in batch:
        seqs.append(item[0])
        feats.append(item[1])

        if len(item) >= 3:
            masks.append(item[2])
        if len(item) == 4:
            labels.append(item[3])

    # ----- PAD SEQUENCES -----
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0)

    # ----- CREATE ATTENTION MASK -----
    # 1 cho token thực, 0 cho padding
    attn_mask = torch.zeros_like(seqs_padded, dtype=torch.float32)
    for i, seq in enumerate(seqs):
        attn_mask[i, :len(seq)] = 1.0

    # ----- STACK FEATURES -----
    feats = torch.stack(feats, dim=0)

    # ----- STACK MASK -----
    masks_tensor = torch.stack(masks, dim=0) if masks else None

    # ----- STACK LABEL -----
    labels_tensor = torch.stack(labels, dim=0) if labels else None

    # ----- RETURN -----
    if masks_tensor is not None and labels_tensor is not None:
        return seqs_padded, feats, masks_tensor, labels_tensor, attn_mask
    if masks_tensor is not None:
        return seqs_padded, feats, masks_tensor, attn_mask
    if labels_tensor is not None:
        return seqs_padded, feats, labels_tensor, attn_mask
    return seqs_padded, feats, attn_mask