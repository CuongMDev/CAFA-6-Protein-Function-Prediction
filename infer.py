from collections import defaultdict, deque
import heapq

import torch
import numpy as np
from tqdm import tqdm
from HybridModel import HybridModel
import torch.nn as nn
from load_data import load_test, info_split
from custom_dataset import CustomDataset
from config import VAL_BATCH_SIZE, test_seq_file, submit_known, embedding_dim, DEVICE


def load_model(model_path, output_dim, input_seq_dim, input_feat_dim, device=DEVICE):
    model = HybridModel(
        output_dim=output_dim, 
        input_seq_dim=input_seq_dim, 
        input_feat_dim=input_feat_dim,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    model = torch.jit.script(model)
    return model


def infer(model, seq_ids, features, device=DEVICE):
    seq_ids = seq_ids.to(device)
    features = features.to(device)

    with torch.no_grad():
        outputs = model(seq_ids, features)

    outputs = torch.sigmoid(outputs)

    # Đưa output về CPU để giải phóng GPU
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

def conditional_to_raw_batch(probs_cond_batch, child_to_parents, topo_order=None, prior_mean=0.02):
    """
    Phiên bản xử lý batch, probs_cond_batch: (batch_size, n_terms)
    """
    probs_raw_batch = np.zeros_like(probs_cond_batch, dtype=float)

    for term in topo_order:
        parents = child_to_parents[term]
        if len(parents) == 0:
            probs_raw_batch[:, term] = probs_cond_batch[:, term]
        else:
            parent_probs = probs_raw_batch[:, parents]  # shape: (batch_size, n_parents)
            p_parent_exist = 1 - np.prod(1 - parent_probs, axis=1)
            probs_raw_batch[:, term] = probs_cond_batch[:, term] * p_parent_exist

    # Các node chưa có parent nào (nếu DAG disconnected)
    # Không cần dùng processed, topo_order đảm bảo đã tính hết các node có parent
    # Các node isolated có topo_order ở đầu hoặc cuối, probs_raw sẽ = probs_cond hoặc prior_mean
    return probs_raw_batch

def create_submission(dataloader, model, entry_ids, term_vocab,
                            parent_to_children,
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

    child_to_parents = build_graph_structures(parent_to_children)

    # --- Tính topo order một lần cho DAG ---
    topo_order = compute_topo_order(child_to_parents, parent_to_children)

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
                buffer_ids.append(protein_id)
                idx += 1

                # Khi buffer đầy
                if len(buffer_probs) >= buffer_size:
                    buffer_probs_np = np.array(buffer_probs)  # (buffer_size, n_terms)

                    # --- Tính conditional_to_raw batch ---
                    consistent_probs_batch = conditional_to_raw_batch(
                        buffer_probs_np, child_to_parents, topo_order=topo_order
                    )

                    # --- Ghi ra file ---
                    for protein_id, consistent_probs, probs_orig in zip(
                            buffer_ids, consistent_probs_batch, buffer_probs_np):
                        total_consistency_boost += np.mean(np.abs(consistent_probs - probs_orig))
                        # Lấy top_k nếu top_k không None, còn không duyệt tất cả
                        top_indices = range(len(consistent_probs))
                        total_expanded += len(top_indices)
                        
                        # print(sum(consistent_probs > 0.1))

                        for t in top_indices:
                            if consistent_probs[t] < threshold:
                                continue
                            term_id = term_vocab[t]
                            prob = consistent_probs[t]
                            write_buffer.append(f"{protein_id}\t{term_id}\t{prob:.6f}\n")
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
                buffer_probs_np, child_to_parents, topo_order=topo_order
            )

            for protein_id, consistent_probs, probs_orig in zip(buffer_ids, consistent_probs_batch, buffer_probs_np):
                total_consistency_boost += np.mean(np.abs(consistent_probs - probs_orig))
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
    test_protein_vocab, amino_axit, ox_features, train_protein_vocab, term_vocab, ox_vocab, graph = load_test()

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
    model = load_model(
        model_path=model_save_path,
        output_dim=len(term_vocab), 
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
        device=DEVICE,
        # top_k=2000,
        threshold=0.01,  # ✅ Tăng threshold để giảm kích thước file (0.3-0.5 là hợp lý)
        output_file=submission_file
    )

    print(f"\n{'=' * 60}")
    print(f"✅ HOÀN THÀNH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")