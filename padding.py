import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# --- Custom Dataset để xử lý padding và one-hot ---
class CustomDataset(Dataset):
    def __init__(self, seq_data, feature_data, labels, num_labels):
        """
        seq_data: list of list of token ids
        feature_data: numpy array (b, linear_input_dim)
        labels: list of list các label
        num_labels: tổng số nhãn (size của one-hot)
        """
        self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in seq_data]
        self.feature_data = torch.tensor(feature_data, dtype=torch.float32)
        self.labels = [self.to_one_hot(lab, num_labels) for lab in labels]

    def to_one_hot(self, label_list, num_labels):
        one_hot = torch.zeros(num_labels, dtype=torch.float32)
        one_hot[label_list] = 1.0
        return one_hot

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        return self.seq_data[idx], self.feature_data[idx], self.labels[idx]

# --- collate_fn để pad trái sequence mỗi batch ---
def collate_fn(batch):

    seqs, features, labels = zip(*batch)
    # pad trái
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0, padding_side="left")  # default pad right, sẽ đổi thành pad trái
    max_len = seqs_padded.size(1)
    seqs_padded = torch.stack([torch.cat([torch.zeros(max_len - len(s), dtype=torch.long), s]) if len(s) < max_len else s for s in seqs])
    features = torch.stack(features)
    labels = torch.stack(labels)
    return seqs_padded, features, labels


import torch

def flexible_collate(batch):
    # kiểm tra batch[0] có labels hay không
    if len(batch[0]) == 3:
        seqs, features, labels = zip(*batch)
        has_labels = False
    elif len(batch[0]) == 2:
        seqs, features = zip(*batch)
        labels = None
        has_labels = False
    else:
        raise ValueError("Batch phải có 2 hoặc 3 phần tử")

    # --- chuyển seqs sang tensor ---
    seqs = [torch.tensor(s, dtype=torch.long) for s in seqs]
    max_len = max(len(s) for s in seqs)
    seqs_padded = torch.stack([
        torch.cat([torch.zeros(max_len - len(s), dtype=torch.long), s])
        for s in seqs
    ])

    # features sang tensor
    features = torch.stack([torch.tensor(f, dtype=torch.float32) for f in features])

    if has_labels:
        labels = torch.stack([torch.tensor(l, dtype=torch.float32) for l in labels])
        return seqs_padded, features, labels
    else:
        return seqs_padded, features
