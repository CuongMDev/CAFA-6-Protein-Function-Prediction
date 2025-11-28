from collections import defaultdict, deque
import heapq

import torch
import numpy as np
from tqdm import tqdm
from HybridModel import HybridModel
import torch.nn as nn
from load_data import load_test, info_split
from custom_dataset import CustomDataset
from config import *

def load_models(model_dir, k_folds, output_dim, input_seq_dim, input_feat_dim, device=DEVICE):
    models = []

    for fold in range(k_folds):
        model_path = f"{model_dir}best_model_fold{fold}.pt"
        model = HybridModel(
            output_dim=output_dim,
            input_seq_dim=input_seq_dim,
            input_feat_dim=input_feat_dim
        )
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()

        models.append(model)

    return models

def infer(models, seq_ids, features, device=DEVICE, ensemble="mean"):
    """
    Ensemble inference bằng nhiều fold models.

    Args:
        models  : list các model đã load weight từ từng fold
        seq_ids : tensor [B, seq_dim]
        features: tensor [B, feat_dim]
        device  : CUDA/CPU
        ensemble: "mean" hoặc "max"

    Returns:
        probs: tensor [B, C] sau sigmoid (ensemble)
    """
    seq_ids = seq_ids.to(device)
    features = features.to(device)

    all_outputs = []

    for model in models:
        model.eval()
        model.to(device)

        with torch.no_grad():
            logits = model(seq_ids, features)  # [B, C]
            probs = torch.sigmoid(logits)      # sigmoid tại đây
            all_outputs.append(probs.cpu())    # đưa về CPU giải phóng VRAM

    # Stack outputs: [K, B, C]
    all_outputs = torch.stack(all_outputs, dim=0)

    if ensemble == "mean":
        final_outputs = all_outputs.mean(dim=0)
    elif ensemble == "max":
        final_outputs = all_outputs.max(dim=0).values
    else:
        raise ValueError("ensemble must be 'mean' or 'max'")

    return final_outputs


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
def build_graph_structures(child_to_parents):
    """
    Xây dựng child->parents chỉ dùng list
    """
    parent_to_children = [[] for _ in range(len(child_to_parents))]

    # lấp dữ liệu
    for child, parents in enumerate(child_to_parents):
        for p in parents:
            parent_to_children[p].append(child)

    return child_to_parents

def compute_topo_order(child_to_parents, parent_to_children):
    """
    Tính topo order của DAG một lần, trả về list các term theo thứ tự
    parent được xử lý trước child.
    """
    indegree = np.array([len(pars) for pars in child_to_parents], dtype=int)
    order = []

    queue = np.where(indegree == 0)[0].tolist()
    while queue:
        term = queue.pop(0)
        order.append(term)
        for child in parent_to_children[term]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return order

def conditional_to_raw_batch(probs_cond_batch, child_to_parents_masked, child_to_parents_full,
                             parent_to_children_full, topo_order_masked, topo_order_full):
    """
    Phiên bản xử lý batch, trả về xác suất cho toàn bộ DAG.

    Args:
        probs_cond_batch: (batch_size, NUM_CLASSES), xác suất model output
        child_to_parents_masked: DAG chỉ gồm node < NUM_CLASSES
        topo_order_masked: topo order chỉ node < NUM_CLASSES
        child_to_parents_full: DAG đầy đủ
        parent_to_children_full: DAG đầy đủ
        NUM_CLASSES: số class model output
        prior_mean: giá trị gán cho các node chưa có xác suất
    Returns:
        probs_full: (batch_size, n_terms), xác suất toàn DAG
    """
    batch_size = probs_cond_batch.shape[0]
    n_terms = len(child_to_parents_full)
    probs_full = np.zeros((batch_size, n_terms), dtype=float)

    # --- 1) Tính xác suất cho NUM_CLASSES đầu theo topo_masked ---
    for term in topo_order_masked:
        parents = child_to_parents_masked[term]
        if len(parents) == 0:
            probs_full[:, term] = probs_cond_batch[:, term]
        else:
            parent_probs = probs_full[:, parents]
            p_parent_exist = 1 - np.prod(1 - parent_probs, axis=1)
            probs_full[:, term] = probs_cond_batch[:, term] * p_parent_exist

    # --- 2) Propagate sang toàn bộ DAG theo topo full ---
    for term in topo_order_full[::-1]:  # đảo ngược topo
        if term < NUM_CLASSES:
            continue
        children = parent_to_children_full[term]
        if children:
            child_probs = probs_full[:, children]
            max_child = np.max(child_probs, axis=1)
            # Nếu term chưa có xác suất, gán max_child
            probs_full[:, term] = np.maximum(probs_full[:, term], max_child)

    return probs_full

def create_submission(dataloader, model, entry_ids, term_vocab, known_protein_to_terms_dict,
                            parent_to_children, parent_to_children_masked,
                            device, threshold=0.0, buffer_size=1000,
                            output_file="submission.tsv"):
    """
    Phiên bản xử lý buffer batch với conditional_to_raw batch
    """
    total_lines = 0
    total_expanded = 0
    total_consistency_boost = 0.0
    idx = 0

    write_buffer = []
    buffer_probs = []
    buffer_ids = []

    child_to_parents_masked  = build_graph_structures(parent_to_children_masked)
    child_to_parent = build_graph_structures(parent_to_children)

    # --- Tính topo order một lần cho DAG ---
    topo_order_masked = compute_topo_order(child_to_parents_masked, parent_to_children_masked)
    topo_order_full = compute_topo_order(child_to_parent, parent_to_children)

    with open(output_file, "w", encoding="utf-8") as f:
        for batch in tqdm(dataloader, desc="Inference", total=len(dataloader)):
            seq_ids = batch[0].to(device)
            features = batch[1].to(device)

            # model output
            outputs = infer(model, seq_ids, features)  # tensor
            outputs = outputs.cpu().numpy()  # (batch_size, n_terms)
            batch_size = outputs.shape[0]

            # Lưu vào buffer
            for j in range(batch_size):
                if idx >= len(entry_ids):
                    break
                protein_id = entry_ids[idx]
                buffer_probs.append(outputs[j])
                buffer_ids.append(idx)
                idx += 1

                # Khi buffer đầy
                if len(buffer_probs) >= buffer_size:
                    buffer_probs_np = np.array(buffer_probs)  # (buffer_size, n_terms)

                    # --- Tính conditional_to_raw batch ---
                    consistent_probs_batch = conditional_to_raw_batch(
                        buffer_probs_np, child_to_parents_masked, child_to_parent, parent_to_children, topo_order_masked=topo_order_masked, topo_order_full=topo_order_full
                    )

                    # --- Ghi ra file ---
                    for protein_idx, consistent_probs, probs_orig in zip(
                            buffer_ids, consistent_probs_batch, buffer_probs_np):
                        total_consistency_boost += np.mean(np.abs(consistent_probs[:NUM_CLASSES] - probs_orig))
                        # Lấy top_k nếu top_k không None, còn không duyệt tất cả
                        top_indices = range(len(consistent_probs))
                        total_expanded += len(top_indices)
                        
                        # print(sum(consistent_probs > 0.1))

                        for t in top_indices:
                            if consistent_probs[t] < threshold:
                                continue
                            term_id = term_vocab[t]
                            if protein_idx in known_protein_to_terms_dict and \
                                term_id in known_protein_to_terms_dict[protein_idx]:
                                continue
                            prob = consistent_probs[t]
                            write_buffer.append(f"{entry_ids[protein_idx]}\t{term_id}\t{prob:.6f}\n")
                            total_lines += 1

                    # Ghi buffer ra file và reset
                    f.writelines(write_buffer)
                    write_buffer = []
                    buffer_probs = []
                    buffer_ids = []

        # --- Xử lý buffer còn lại sau khi hết dataloader ---
        if buffer_probs:
            buffer_probs_np = np.array(buffer_probs)
            consistent_probs_batch = conditional_to_raw_batch(
                buffer_probs_np, child_to_parents_masked, child_to_parent, parent_to_children, topo_order_masked=topo_order_masked, topo_order_full=topo_order_full
            )

            for protein_id, consistent_probs, probs_orig in zip(buffer_ids, consistent_probs_batch, buffer_probs_np):
                total_consistency_boost += np.mean(np.abs(consistent_probs[:NUM_CLASSES] - probs_orig))
                top_indices = range(len(consistent_probs))
                total_expanded += len(top_indices)
                for t in top_indices:
                    if consistent_probs[t] < threshold:
                        continue
                    term_id = term_vocab[t]
                    prob = consistent_probs[t]
                    write_buffer.append(f"{protein_id}\t{term_id}\t{prob:.6f}\n")
                    total_lines += 1

            f.writelines(write_buffer)

        with open(submit_known, "r", encoding="utf-8") as sf:
            for line in sf:
                f.write(line)

    print(f"\n{'='*60}")
    print(f"✅ Submission saved: {output_file}")
    print(f"📊 Total lines: {total_lines:,}")
    print(f"🌳 Total expanded terms: {total_expanded:,}")
    print(f"🔧 Avg consistency boost: {total_consistency_boost / len(entry_ids):.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    from config import DEVICE, model_save_path
    from torch.utils.data import DataLoader
    import os
    from datetime import datetime

    print(f"\n{'=' * 60}")
    print(f"🚀 BẮT ĐẦU INFERENCE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # Load test data
    test_protein_vocab, amino_axit, ox_features, train_protein_vocab, term_vocab, ox_vocab, graph, graph_masked, protein_to_terms_dict = load_test()

    print(f"🔍 Debug - ox_features type: {type(ox_features)}")
    print(f"🔍 Debug - ox_features shape: {ox_features.shape if hasattr(ox_features, 'shape') else len(ox_features)}")
    print(f"🔍 Debug - amino_axit length: {len(amino_axit)}")
    print(f"🔍 Debug - ox_vocab length: {len(ox_vocab)}")

    print(f"🔍 Debug - graph type: {type(graph)}")
    print(f"🔍 Debug - graph length: {len(graph)}")

    # Tạo dataset và dataloader
    dataset = CustomDataset(amino_axit, ox_features)
    # dataset = torch.utils.data.Subset(dataset, list(range(10)))
    dataloader = DataLoader(dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    # Khởi tạo model với vocab từ TRAIN (vì model được train với vocab này)
    model = load_models(
        model_dir=model_dir,
        k_folds=k_folds,
        output_dim=top_k, 
        input_seq_dim=embedding_dim, 
        input_feat_dim=len(ox_vocab),
        device=DEVICE
    )

    # Lấy EntryIDs từ test_protein_vocab (đúng thứ tự với dataloader)
    entry_ids = test_protein_vocab
    print(f"📊 Tổng số protein trong test: {len(entry_ids)}")
    print(f"📊 Tổng số protein trong train vocab: {len(train_protein_vocab)}")
    print(f"📊 Số samples trong dataloader: {len(dataset)}")

    # Kiểm tra file submission đã tồn tại chưa
    submission_file = f"submission.tsv"

    # Tạo submission với winning solution postprocessing
    print(f"\n🚀 Bắt đầu tạo submission...")
    create_submission(
        dataloader=dataloader,
        model=model,
        entry_ids=entry_ids,
        term_vocab=term_vocab,
        parent_to_children=graph,  # Dùng graph structure đúng
        parent_to_children_masked=graph_masked,
        device=DEVICE,
        known_protein_to_terms_dict=protein_to_terms_dict,
        # top_k=2000,
        threshold=0.01,  # ✅ Tăng threshold để giảm kích thước file (0.3-0.5 là hợp lý)
        output_file=submission_file
    )

    print(f"\n{'=' * 60}")
    print(f"✅ HOÀN THÀNH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")