import torch
import numpy as np
from HybridModel import HybridModel
import torch.nn as nn
from load_data import load_data, info_split
from padding import collate_fn, flexible_collate
from config import train_seq_file, lstm_hidden, lstm_layers, linear_hidden_dim, classifier_hidden_dim


def load_model(model_path, num_labels, vocab_size, linear_input_dim, device,
               embedding_dim=64, lstm_hidden=128, lstm_layers=1,
               linear_hidden_dim=64, classifier_hidden_dim=128):
    model = HybridModel(
        num_labels=num_labels,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        linear_input_dim=linear_input_dim,
        linear_hidden_dim=linear_hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def infer(model, seq_ids, features, device):
    seq_ids = seq_ids.to(device)
    features = features.to(device)
    with torch.no_grad():
        outputs = model(seq_ids, features)
    
    
    outputs = torch.sigmoid(outputs)
    return outputs

def extract_entry_ids(fasta_file):
    """Trích xuất EntryIDs từ file FASTA theo thứ tự"""
    entry_ids = []
    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                entry_id, _, _, _ = info_split(line)
                if entry_id:
                    entry_ids.append(entry_id)
    return entry_ids

def create_submission(outputs_list, entry_ids, term_vocab, threshold=0.01, output_file="submission.tsv"):
    """
    Tạo file submission từ outputs
    
    Args:
        outputs_list: list các tensor outputs từ các batch
        entry_ids: list các protein IDs
        term_vocab: list các term IDs
        threshold: ngưỡng xác suất (mặc định 0.01)
        output_file: tên file output
    """
    # Gộp tất cả outputs lại
    all_outputs = torch.cat(outputs_list, dim=0).cpu().numpy()
    
    # Kiểm tra số lượng
    num_outputs = all_outputs.shape[0]
    num_entry_ids = len(entry_ids)
    
    if num_outputs != num_entry_ids:
        print(f"Cảnh báo: Số lượng outputs ({num_outputs}) khác với số lượng entry_ids ({num_entry_ids})")
        # Lấy số lượng nhỏ hơn để tránh lỗi
        min_len = min(num_outputs, num_entry_ids)
        all_outputs = all_outputs[:min_len]
        entry_ids = entry_ids[:min_len]
        print(f"Sử dụng {min_len} mẫu đầu tiên")
    
    # Tạo file submission
    total_lines = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for i, protein_id in enumerate(entry_ids):
            # Lấy xác suất cho protein thứ i
            probs = all_outputs[i]
            
            # Tìm các term có xác suất > threshold
            term_indices = np.where(probs > threshold)[0]
            
            # Ghi vào file
            for term_idx in term_indices:
                term_id = term_vocab[term_idx]
                prob = probs[term_idx]
                f.write(f"{protein_id}\t{term_id}\t{prob:.6f}\n")
                total_lines += 1
    
    print(f"Đã tạo file submission: {output_file}")
    print(f"Tổng số dòng: {total_lines}")

if __name__ == "__main__":
    from config import DEVICE, model_save_path
    from torch.utils.data import DataLoader, TensorDataset

    seq_data, feature_data, labels, protein_vocab, term_vocab, ox_vocab = load_data()

    # Tạo dataset dưới dạng list of tuples
    dataset = list(zip(seq_data, feature_data, labels))


    # dataset = TensorDataset(seq_data, feature_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=flexible_collate)

    # num_labels = len(term_vocab)
    # vocab_size = len(protein_vocab)
    # linear_input_dim = len(ox_vocab)

    # model = load_model(model_save_path, num_labels=10, vocab_size=20, linear_input_dim=150, device=DEVICE)

    model = HybridModel(
    num_labels=len(term_vocab), 
        vocab_size=len(protein_vocab), 
        linear_input_dim=len(ox_vocab), 
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        linear_hidden_dim=linear_hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim,
    )
    model.to(DEVICE)
    model.eval()

    # Lấy EntryIDs từ file FASTA
    entry_ids = extract_entry_ids(train_seq_file)
    print(f"Tổng số protein: {len(entry_ids)}")
    
    # Xử lý tất cả các batch
    # all_outputs = []
    # for batch_idx, (seq_ids, features) in enumerate(dataloader):
    #     outputs = infer(model, seq_ids, features, device=DEVICE)
    #     all_outputs.append(outputs)
    #     if (batch_idx + 1) % 100 == 0:
    #         print(f"Đã xử lý {batch_idx + 1} batches...")
    
    # print(f"Tổng số batches: {len(all_outputs)}")

    # Lấy 1 batch đầu tiên
    seq_ids, features = next(iter(dataloader))  # flexible_collate trả về 2 hoặc 3 giá trị
    outputs = infer(model, seq_ids, features, device=DEVICE)
    print(outputs.shape)   # in shape để kiểm tra
    print(outputs[:5])     # in 5 mẫu đầu tiên
    print(outputs)
    
    # Tạo file submission
    create_submission([outputs], entry_ids, term_vocab, threshold=0.01, output_file="submission.tsv")
