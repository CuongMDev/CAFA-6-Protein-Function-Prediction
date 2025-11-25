from collections import defaultdict, deque

import torch
import numpy as np
from HybridModel import HybridModel
import torch.nn as nn
from load_data import load_test, info_split
from custom_dataset import CustomDataset, collate_fn
from config import VAL_BATCH_SIZE, test_seq_file, transformer_hidden, transformer_layers, linear_hidden_dim, nhead, \
    classifier_hidden_dim


def load_model(model_path, num_labels, vocab_size, linear_input_dim, device, nhead=8,
               embedding_dim=64, transformer_hidden=128, transformer_layers=1,
               linear_hidden_dim=32, classifier_hidden_dim=64):
    model = HybridModel(
        num_labels=num_labels,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        nhead=nhead,
        transformer_hidden=transformer_hidden,
        transformer_layers=transformer_layers,
        linear_input_dim=linear_input_dim,
        linear_hidden_dim=linear_hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    model = torch.jit.script(model)
    return model


def infer(model, seq_ids, features, attn_mask=None, device="cpu"):
    try:
        seq_ids = seq_ids.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(seq_ids, features, attn_mask)

        outputs = torch.sigmoid(outputs)

        # Đưa output về CPU để giải phóng GPU
        return outputs.cpu()

    except torch.cuda.OutOfMemoryError:
        print("GPU OOM → chuyển sang CPU cho batch này")
        torch.cuda.empty_cache()

        # chạy lại bằng CPU
        seq_ids = seq_ids.cpu()
        features = features.cpu()
        attn_mask = attn_mask.cpu() if attn_mask is not None else None
        model.cpu()

        with torch.no_grad():
            outputs = model(seq_ids, features, attn_mask)

        outputs = torch.sigmoid(outputs)
        return outputs.cpu()


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


# ----------------------------
# Tree-based postprocessing (Winning Solution Style)
# ----------------------------
def build_graph_structures(mask_graph):
    """
    Xây dựng cả parent->children và children->parents từ mask_graph

    Args:
        mask_graph: list of lists, mask_graph[child] = [parent1, parent2, ...]

    Returns:
        parent_to_children: dict mapping parent -> list of children
        child_to_parents: dict mapping child -> list of parents
    """
    parent_to_children = defaultdict(list)
    child_to_parents = defaultdict(list)

    for child, parents in enumerate(mask_graph):
        child_to_parents[child] = list(parents)
        for p in parents:
            parent_to_children[p].append(child)

    return parent_to_children, child_to_parents


def conditional_to_raw(probs_cond, child_to_parents, parent_to_children, prior_mean=0.02):
    """
    Chuyển xác suất điều kiện sang xác suất raw dựa trên DAG.
    probs_cond: numpy array (n_terms,), xác suất điều kiện (P(term|at least one parent exists))
    child_to_parents, parent_to_children: dict DAG
    prior_mean: xác suất dùng cho term chưa từng training (nếu parent chưa có prob)

    Returns:
        probs_raw: numpy array (n_terms,) xác suất raw cuối cùng
    """
    n_terms = len(probs_cond)
    probs_raw = np.zeros(n_terms)

    # 1. Tính topological order để chắc chắn parent được xử lý trước child
    indegree = {i: 0 for i in range(n_terms)}
    for child, parents in child_to_parents.items():
        indegree[child] = len(parents)

    queue = deque([i for i in range(n_terms) if indegree[i] == 0])
    processed = set()

    while queue:
        term = queue.popleft()
        processed.add(term)

        # Lấy xác suất parent raw
        parents = child_to_parents.get(term, [])
        if parents:
            parent_probs = [probs_raw[p] for p in parents]
        else:
            parent_probs = []

        # Nếu không có parent, xác suất raw = xác suất condition
        if not parent_probs:
            probs_raw[term] = probs_cond[term]
        else:
            # Giả sử các parent độc lập
            p_parent_exist = 1 - np.prod([1 - p for p in parent_probs])
            probs_raw[term] = probs_cond[term] * p_parent_exist

        # Thêm các child của term vào queue nếu đã xử lý đủ parent
        for child in parent_to_children.get(term, []):
            if child in processed:
                continue
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    # Nếu còn term nào chưa xử lý (vì DAG có disconnected node), gán prior_mean
    for i in range(n_terms):
        if i not in processed:
            probs_raw[i] = prior_mean
            print(f"Term {i} not processed")

    return probs_raw

def expand_predictions(pred_terms, parent_to_children, child_to_parents):
    """
    Expand predictions theo propagation rule:
    - Nếu term được predict, tất cả parents phải được thêm vào
    - Có thể thêm children nếu muốn (optional)

    Args:
        pred_terms: set of predicted term indices
        parent_to_children: dict
        child_to_parents: dict

    Returns:
        expanded: set of term indices sau khi expand
    """
    expanded = set(pred_terms)

    # Thêm tất cả parents (mandatory theo propagation rule)
    queue = list(pred_terms)
    while queue:
        current = queue.pop(0)
        parents = child_to_parents.get(current, [])
        for parent in parents:
            if parent not in expanded:
                expanded.add(parent)
                queue.append(parent)

    return expanded


def create_submission(dataloader, model, entry_ids, term_vocab, mask_graph,
                      device, threshold, output_file="submission.tsv"):
    """
    Tạo file submission với winning solution postprocessing:
    1. Consistent postprocessing (average of original + propagated probs)
    2. Expand predictions theo propagation rule
    3. Chỉ lọc threshold ở bước cuối
    """
    from tqdm import tqdm

    # Xây dựng graph structures
    parent_to_children, child_to_parents = build_graph_structures(mask_graph)

    total_lines = 0
    total_expanded = 0
    total_consistency_boost = 0
    idx = 0

    # Buffer để ghi batch
    write_buffer = []
    buffer_size = 10000

    with open(output_file, "w", encoding="utf-8") as f:
        for batch in tqdm(dataloader, desc="Inference", total=len(dataloader)):

            seq_ids = batch[0].to(device)
            features = batch[1].to(device)
            attn_mask = batch[-1].to(device)

            outputs = infer(model, seq_ids, features, attn_mask, device=device)
            outputs = outputs.cpu().numpy()

            batch_size = outputs.shape[0]
            for j in range(batch_size):
                if idx >= len(entry_ids):
                    break

                protein_id = entry_ids[idx]
                probs = outputs[j]

                # Step 1: Consistent postprocessing
                consistent_probs = conditional_to_raw(
                    probs, parent_to_children, child_to_parents
                )

                total_consistency_boost += np.mean(
                    np.abs(consistent_probs - probs)
                )

                # Step 2: Lấy hết terms (không dùng threshold)
                term_indices = list(range(len(consistent_probs)))
                original_count = len(term_indices)

                # Step 3: Expand
                expanded_indices = expand_predictions(
                    set(term_indices),
                    parent_to_children,
                    child_to_parents
                )
                total_expanded += len(expanded_indices) - original_count

                # ⛔ Step 4: Chỉ lọc threshold ngay trước khi ghi
                for t in expanded_indices:
                    if consistent_probs[t] < threshold:
                        continue  # term < threshold → bỏ

                    term_id = term_vocab[t]
                    prob = consistent_probs[t]
                    write_buffer.append(f"{protein_id}\t{term_id}\t{prob:.6f}\n")
                    total_lines += 1

                # Ghi buffer nếu đủ lớn
                if len(write_buffer) >= buffer_size:
                    f.writelines(write_buffer)
                    write_buffer = []

                idx += 1

        # Ghi phần còn lại
        if write_buffer:
            f.writelines(write_buffer)

    print(f"\n{'=' * 60}")
    print(f"✅ Đã tạo file submission: {output_file}")
    print(f"{'=' * 60}")
    print(f"📊 Tổng số dòng: {total_lines:,}")
    print(f"🌳 Số terms expand thêm: {total_expanded:,}")
    print(f"🔧 Avg consistency boost: {total_consistency_boost / len(entry_ids):.6f}")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    from config import DEVICE, model_save_path
    from torch.utils.data import DataLoader
    import os
    from datetime import datetime

    print(f"\n{'=' * 60}")
    print(f"🚀 BẮT ĐẦU INFERENCE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # Load test data
    test_protein_vocab, amino_axit, ox_features, train_protein_vocab, term_vocab, ox_vocab, graph = load_test()

    # FIX: load_test() có bug - ox_onehot không được unpack đúng
    if isinstance(ox_features, tuple):
        print("⚠️  Phát hiện ox_features là tuple, đang unpack...")
        ox_features = ox_features[0]

    print(f"🔍 Debug - ox_features type: {type(ox_features)}")
    print(f"🔍 Debug - ox_features shape: {ox_features.shape if hasattr(ox_features, 'shape') else len(ox_features)}")
    print(f"🔍 Debug - amino_axit length: {len(amino_axit)}")
    print(f"🔍 Debug - ox_vocab length: {len(ox_vocab)}")

    print(f"🔍 Debug - graph type: {type(graph)}")
    print(f"🔍 Debug - graph length: {len(graph)}")

    # Tạo dataset và dataloader
    dataset = CustomDataset(amino_axit, ox_features)
    # dataset = torch.utils.data.Subset(dataset, list(range(10)))
    dataloader = DataLoader(dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo model với vocab từ TRAIN (vì model được train với vocab này)
    model = load_model(
        model_path=model_save_path,
        num_labels=len(term_vocab), 
        vocab_size=len(train_protein_vocab), 
        transformer_hidden=transformer_hidden,
        transformer_layers=transformer_layers,
        nhead=nhead,
        linear_input_dim=len(ox_vocab), 
        linear_hidden_dim=linear_hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim,
        device=DEVICE
    )

    # Lấy EntryIDs từ test_protein_vocab (đúng thứ tự với dataloader)
    entry_ids = test_protein_vocab
    print(f"📊 Tổng số protein trong test: {len(entry_ids)}")
    print(f"📊 Tổng số protein trong train vocab: {len(train_protein_vocab)}")
    print(f"📊 Số samples trong dataloader: {len(dataset)}")

    # Kiểm tra file submission đã tồn tại chưa
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_file = f"submission_{timestamp}.tsv"

    print(f"\n📝 File submission sẽ được tạo: {submission_file}")

    # Tạo submission với winning solution postprocessing
    print(f"\n🚀 Bắt đầu tạo submission...")
    create_submission(
        dataloader=dataloader,
        model=model,
        entry_ids=entry_ids,
        term_vocab=term_vocab,
        mask_graph=graph,  # Dùng graph structure đúng
        device=DEVICE,
        threshold=0.1,  # ✅ Tăng threshold để giảm kích thước file (0.3-0.5 là hợp lý)
        output_file=submission_file
    )

    print(f"\n{'=' * 60}")
    print(f"✅ HOÀN THÀNH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")