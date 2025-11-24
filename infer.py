from collections import defaultdict

import torch
import numpy as np
from HybridModel import HybridModel
import torch.nn as nn
from load_data import load_test, info_split
from custom_dataset import CustomDataset, collate_fn
from config import BATCH_SIZE, test_seq_file, transformer_hidden, transformer_layers, linear_hidden_dim, \
    classifier_hidden_dim


def load_model(model_path, num_labels, vocab_size, linear_input_dim, device,
               embedding_dim=64, transformer_hidden=128, transformer_layers=1,
               linear_hidden_dim=32, classifier_hidden_dim=64):
    model = HybridModel(
        num_labels=num_labels,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
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
    return model


def infer(model, seq_ids, features, attn_mask=None, device="cpu"):
    try:
        seq_ids = seq_ids.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device, non_blocking=True)

        model.eval()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
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


def propagate_to_parents(probs, child_to_parents):
    """
    Propagate probabilities UP: nếu child có prob cao, parent phải có prob >= child
    Theo propagation rule: nếu term exists, tất cả parents phải exist

    Args:
        probs: numpy array of shape (n_terms,)
        child_to_parents: dict mapping child -> list of parents

    Returns:
        propagated_probs: numpy array sau khi propagate lên parents
    """
    propagated = probs.copy()

    # Lặp ít hơn và chỉ xét terms có prob > 0
    max_iterations = 5  # Giảm từ 10 xuống 5
    for _ in range(max_iterations):
        changed = False
        for child, parents in child_to_parents.items():
            child_prob = propagated[child]
            if child_prob > 0:  # Chỉ xét nếu child có prob
                for parent in parents:
                    if propagated[parent] < child_prob:
                        propagated[parent] = child_prob
                        changed = True

        if not changed:
            break

    return propagated


def propagate_to_children(probs, parent_to_children):
    """
    Propagate probabilities DOWN: lấy max prob từ children propagate ngược lên

    Args:
        probs: numpy array of shape (n_terms,)
        parent_to_children: dict mapping parent -> list of children

    Returns:
        propagated_probs: numpy array với max children probs
    """
    propagated = probs.copy()

    # Lặp từ leaf lên root (tối ưu)
    max_iterations = 5  # Giảm từ 10 xuống 5
    for _ in range(max_iterations):
        changed = False
        for parent, children in parent_to_children.items():
            if children:
                max_child_prob = max(propagated[c] for c in children)
                if max_child_prob > propagated[parent]:
                    propagated[parent] = max_child_prob
                    changed = True

        if not changed:
            break

    return propagated


def consistent_postprocessing(probs, parent_to_children, child_to_parents):
    """
    Áp dụng consistent postprocessing như trong winning solution:
    Final prediction = average of:
      - Original term probability
      - Maximum propagated children probability
      - Minimum propagated parents probability (actually max, theo propagation rule)

    Args:
        probs: numpy array of shape (n_terms,)
        parent_to_children: dict
        child_to_parents: dict

    Returns:
        consistent_probs: numpy array sau khi làm consistent
    """
    # 1. Original probabilities
    original = probs.copy()

    # 2. Propagate UP (children -> parents): parents phải >= max(children)
    from_children = propagate_to_children(probs, parent_to_children)

    # 3. Propagate DOWN (parents -> children): đảm bảo consistency
    # Nếu parent có prob thấp, children không thể có prob cao hơn
    from_parents = propagate_to_parents(probs, child_to_parents)

    # 4. Average cả 3 (theo paper: boosts score a little)
    consistent = (original + from_children + from_parents) / 3.0

    return consistent


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
                      device, threshold=0.3, output_file="submission.tsv",
                      max_terms_per_protein=200):
    """
    Tạo file submission với winning solution postprocessing:
    1. Consistent postprocessing (average of original + propagated probs)
    2. Expand predictions theo propagation rule

    Args:
        dataloader: DataLoader chứa data
        model: model đã train
        entry_ids: list các protein IDs
        term_vocab: list các term IDs
        mask_graph: cấu trúc cây GO terms (list of list parents)
        device: cuda hoặc cpu
        threshold: ngưỡng xác suất (khuyến nghị: 0.3-0.5)
        output_file: tên file output
        max_terms_per_protein: số lượng terms tối đa cho mỗi protein (giới hạn để tránh file quá lớn)
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
    buffer_size = 10000  # Ghi mỗi 10k dòng

    with open(output_file, "w", encoding="utf-8") as f:
        for batch in tqdm(dataloader, desc="Inference", total=len(dataloader)):
            seq_ids = batch[0].to(device)
            features = batch[1].to(device)
            attn_mask = batch[-1].to(device)

            # Infer
            outputs = infer(model, seq_ids, features, attn_mask, device=device)
            outputs = outputs.cpu().numpy()

            # Xử lý từng sample trong batch
            batch_size = outputs.shape[0]
            for j in range(batch_size):
                if idx >= len(entry_ids):
                    break

                protein_id = entry_ids[idx]
                probs = outputs[j]

                # Step 1: Apply consistent postprocessing
                consistent_probs = consistent_postprocessing(
                    probs, parent_to_children, child_to_parents
                )

                # Đo lường sự thay đổi
                prob_change = np.mean(np.abs(consistent_probs - probs))
                total_consistency_boost += prob_change

                # Step 2: Lấy các term > threshold
                term_indices = np.where(consistent_probs > threshold)[0].tolist()
                original_count = len(term_indices)

                # Step 3: Expand theo propagation rule (thêm parents)
                expanded_indices = expand_predictions(
                    set(term_indices), parent_to_children, child_to_parents
                )
                expanded_count = len(expanded_indices) - original_count
                total_expanded += expanded_count

                # Step 4: Giới hạn số terms nếu quá nhiều (lấy top prob)
                if len(expanded_indices) > max_terms_per_protein:
                    # Sắp xếp theo probability giảm dần và lấy top
                    sorted_indices = sorted(
                        expanded_indices,
                        key=lambda t: consistent_probs[t],
                        reverse=True
                    )[:max_terms_per_protein]
                    expanded_indices = set(sorted_indices)

                # Step 5: Thêm vào buffer thay vì ghi trực tiếp
                for t in expanded_indices:
                    term_id = term_vocab[t]
                    prob = consistent_probs[t]
                    write_buffer.append(f"{protein_id}\t{term_id}\t{prob:.6f}\n")
                    total_lines += 1

                # Ghi buffer khi đủ lớn
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
    print(f"🌳 Số terms được expand (parents): {total_expanded:,}")
    print(f"🔧 Avg consistency boost: {total_consistency_boost / len(entry_ids):.6f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    from config import DEVICE, model_save_path
    from torch.utils.data import DataLoader
    from load_data import load_train, parse_obo, build_is_a_graph
    import os
    from datetime import datetime

    print(f"\n{'=' * 60}")
    print(f"🚀 BẮT ĐẦU INFERENCE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # Load test data
    test_protein_vocab, amino_axit, ox_features, _, term_vocab, ox_vocab = load_test()

    # FIX: load_test() có bug - ox_onehot không được unpack đúng
    if isinstance(ox_features, tuple):
        print("⚠️  Phát hiện ox_features là tuple, đang unpack...")
        ox_features = ox_features[0]

    print(f"🔍 Debug - ox_features type: {type(ox_features)}")
    print(f"🔍 Debug - ox_features shape: {ox_features.shape if hasattr(ox_features, 'shape') else len(ox_features)}")
    print(f"🔍 Debug - amino_axit length: {len(amino_axit)}")
    print(f"🔍 Debug - ox_vocab length: {len(ox_vocab)}")

    # Load vocab và mask_graph từ train data
    # mask_graph từ load_train() là function, không phải graph structure!
    _, _, _, _, train_protein_vocab, _, _ = load_train()

    # Load GO terms và build is_a graph
    from config import obo_file

    _, terms = parse_obo(obo_file)
    term2idx = {t['id']: i for i, t in enumerate(terms)}
    graph = build_is_a_graph(terms, term2idx)  # Đây mới là graph structure đúng!

    print(f"🔍 Debug - graph type: {type(graph)}")
    print(f"🔍 Debug - graph length: {len(graph)}")

    # Tạo dataset và dataloader
    dataset = CustomDataset(amino_axit, ox_features)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo model với vocab từ TRAIN (vì model được train với vocab này)
    model = HybridModel(
        num_labels=len(term_vocab),
        vocab_size=len(train_protein_vocab),  # Dùng train vocab
        embedding_dim=128,
        transformer_hidden=512,
        transformer_layers=2,
        linear_input_dim=len(ox_vocab),
        linear_hidden_dim=linear_hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim
    )

    # Load trained weights (nếu có)
    if os.path.exists(model_save_path):
        print(f"📥 Đang load model từ: {model_save_path}")
        state_dict = torch.load(model_save_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Load model thành công!")
    else:
        print(f"⚠️  Không tìm thấy model tại: {model_save_path}")
        print("⚠️  Sẽ sử dụng model chưa train (random weights) - KẾT QUẢ SẼ KHÔNG CHÍNH XÁC!")
        response = input("Bạn có muốn tiếp tục? (y/n): ")
        if response.lower() != 'y':
            print("❌ Dừng chương trình.")
            exit()

    model.to(DEVICE)
    model.eval()

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
        threshold=0.3,  # ✅ Tăng threshold để giảm kích thước file (0.3-0.5 là hợp lý)
        output_file=submission_file
    )

    print(f"\n{'=' * 60}")
    print(f"✅ HOÀN THÀNH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # Khởi tạo model với vocab từ TRAIN (vì model được train với vocab này)
    model = HybridModel(
        num_labels=len(term_vocab),
        vocab_size=len(train_protein_vocab),  # Dùng train vocab
        embedding_dim=128,
        transformer_hidden=512,
        transformer_layers=2,
        linear_input_dim=len(ox_vocab),
        linear_hidden_dim=linear_hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim
    )

    # Load trained weights
    state_dict = torch.load(model_save_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    # Lấy EntryIDs từ test_protein_vocab (đúng thứ tự với dataloader)
    entry_ids = test_protein_vocab
    print(f"📊 Tổng số protein trong test: {len(entry_ids)}")
    print(f"📊 Tổng số protein trong train vocab: {len(train_protein_vocab)}")
    print(f"📊 Số samples trong dataloader: {len(dataset)}")

    # Tạo submission với winning solution postprocessing
    create_submission(
        dataloader=dataloader,
        model=model,
        entry_ids=entry_ids,
        term_vocab=term_vocab,
        mask_graph=mask_graph,
        device=DEVICE,
        threshold=0.01,
        output_file="submission.tsv"
    )