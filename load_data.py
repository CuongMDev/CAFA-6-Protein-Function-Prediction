from collections import Counter, defaultdict
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import *
from functools import lru_cache
import re
import matplotlib.pyplot as plt
from utils import one_hot


def parse_obo(path, weight_path=None):
    """
    Đọc file OBO và (tùy chọn) gắn weight từ IA.tsv.
    Chỉ giữ các term có weight > 0.

    Trả về:
        - terms: list[dict]     → các term trong OBO
        - term_weights: dict    → {term_id: float}, chỉ weight > 0
        - terms_name: list[str] → danh sách ID theo đúng thứ tự xuất hiện, chỉ weight > 0
    """
    terms = []
    term = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                term = {}
                continue
            if line == "" and term is not None:
                if "id" in term:
                    terms.append(term)
                term = None
                continue
            if term is None:
                continue
            if ": " in line:
                key, value = line.split(": ", 1)
                if key in ["synonym", "is_a"]:
                    term.setdefault(key, []).append(value)
                else:
                    term[key] = value

    # Danh sách term name theo thứ tự
    terms_name_all = [t["id"] for t in terms]

    # --------------------------
    # Đọc IA weights nếu có
    # --------------------------
    weights = {}
    if weight_path is not None:
        with open(weight_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                go_id, w = line.split("\t")
                weights[go_id] = float(w)

    # Tạo dict weight, mặc định 0.0 nếu không có trong IA
    term_weights_all = {tid: weights.get(tid, 0.0) for tid in terms_name_all}

    # Chỉ giữ các term có weight > 0
    valid_terms_name = [tid for tid in terms_name_all if term_weights_all[tid] > 0]
    term_weights = {tid: term_weights_all[tid] for tid in valid_terms_name}

    return valid_terms_name, terms, term_weights

def get_emb_amino(emb_file):
    """
    Load embeddings đã lưu từ file .npy.

    Args:
        emb_file (str): đường dẫn file .npy chứa embeddings

    Returns:
        np.ndarray: embeddings (num_sequences, embedding_dim)
    """
    if not os.path.exists(emb_file):
        raise FileNotFoundError(f"{emb_file} không tồn tại")
    
    embeds = np.load(emb_file)
    print(f"Loaded embeddings of shape {embeds.shape} from {emb_file}")
    return embeds

def get_ids_numpy(terms):
    """
       Trả về array chứa ID của tất cả terms.

       Args:
           terms (list[dict])

       Returns:
           np.ndarray: mảng chứa các term id
       """
    return np.array([t["id"] for t in terms])

def get_terms_per_entryid(df, terms_name, weight, top_k, top_k_type):
    """
    Lấy các term cho mỗi EntryID dạng hot-index, áp dụng top_k = [k1, k2, k3].
    Thứ tự aspect được xác định bởi top_k_type.
    
    Args:
        df: DataFrame chứa cột ['EntryID', 'term', 'aspect']
        terms_name: list tất cả term theo thứ tự chuẩn
        weight: dict {term: weight}
        top_k: [k1, k2, k3], số lượng nhãn cần giữ theo aspect
        top_k_type: ['C','F','P'] hoặc thứ tự nào khác, cùng chiều với top_k

    Returns:
        terms_per_entryid_hot: list[list[int]] của mỗi entry, các term đã chuyển sang index
        term2idx: dict term -> index
        term_weights_list: list weight theo thứ tự filtered_terms_list
        filtered_terms_list: list các term đã lọc theo top_k theo top_k_type
    """
    if len(top_k) != 3 or len(top_k_type) != 3:
        raise ValueError("top_k và top_k_type phải có độ dài 3")

    k1, k2, k3 = top_k
    entryids = df["EntryID"].unique()

    # 1) Chỉ giữ các term có weight > 0
    valid_terms = set(t for t in terms_name if weight.get(t, 0.0) > 0)

    # 2) Gom term theo aspect + tần suất
    term_counter = Counter(df["term"].values)
    aspect_dict = dict(zip(df["term"], df["aspect"]))

    aspect_terms = {top_k_type[0]: [], top_k_type[1]: [], top_k_type[2]: []}
    k_dict = {top_k_type[0]: k1, top_k_type[1]: k2, top_k_type[2]: k3}

    for term, _ in term_counter.most_common():
        if term not in valid_terms:
            continue
        asp = aspect_dict.get(term, None)
        if asp in aspect_terms and len(aspect_terms[asp]) < k_dict[asp]:
            aspect_terms[asp].append(term)
        # Dừng nếu đủ tất cả
        if all(len(aspect_terms[a]) == k_dict[a] for a in top_k_type):
            break

    # 3) Gộp theo thứ tự top_k_type
    filtered_terms_list = []
    for a in top_k_type:
        filtered_terms_list.extend(aspect_terms[a])
    most_common_terms = set(filtered_terms_list)

    # 4) EntryID -> list term
    entryid_term_dict = {}
    for eid, term in df[["EntryID", "term"]].values:
        if term in most_common_terms:
            entryid_term_dict.setdefault(eid, set()).add(term)

    terms_per_entryid = [list(entryid_term_dict.get(eid, set())) for eid in entryids]

    # 5) Chuyển sang index
    term2idx = {t: i for i, t in enumerate(filtered_terms_list)}
    terms_per_entryid = [
        [term2idx[label] for label in sublist if label in term2idx]
        for sublist in terms_per_entryid
    ]

    # 6) Danh sách trọng số theo thứ tự filtered_terms_list
    term_weights_list = [weight[t] for t in filtered_terms_list]

    return terms_per_entryid, term2idx, term_weights_list, filtered_terms_list
def term_protein_counts(df, terms):
    """
    Trả về số lượng protein mà mỗi term xuất hiện.

    Args:
        df: DataFrame có cột ["EntryID", "term"]
        terms: list dict, mỗi dict có key 'id' của term

    Returns:
        dict: term_id -> số protein có term đó
    """
    # Lấy danh sách term muốn xét
    terms_name = [t['id'] for t in terms]

    # Lọc DataFrame chỉ giữ các term trong terms_name
    filtered = df[df["term"].isin(terms_name)]

    # Tạo dict: term -> set protein
    term_protein_dict = {}
    for eid, term in filtered[["EntryID", "term"]].values:
        term_protein_dict.setdefault(term, set()).add(eid)

    # Chuyển set protein thành số lượng
    term_counts = [len(proteins) for term, proteins in term_protein_dict.items()]

    return term_counts


def build_is_a_graph(terms, term2idx, terms_per_entryid_hot):
    """
    Build is_a graph và propagate labels cho mỗi EntryID thành list index đã mở rộng.

    Args:
        terms: danh sách dict term, có key "id" và "is_a"
        term2idx: dict term -> index
        terms_per_entryid_hot: list[list[int]] ban đầu, chưa propagate

    Returns:
        graph: list[list[int]], graph[idx] = list parent idx
        labels_propagated: list[list[int]], mỗi list là index đã bao gồm parent
    """
    # 1) Build graph
    temp_parents = {}
    for t in terms:
        tid = t["id"]
        temp_parents[tid] = [x.split(" ! ")[0] for x in t.get("is_a", [])]

    memo = {}

    def get_real_parents(tid):
        if tid not in temp_parents:
            return set()
        if tid in memo:
            return memo[tid]

        real_parents = set()
        for p in temp_parents.get(tid, []):
            if p in term2idx:
                real_parents.add(term2idx[p])
            else:
                real_parents |= get_real_parents(p)
        memo[tid] = real_parents
        return real_parents

    n_terms = len(term2idx)
    graph = [[] for _ in range(n_terms)]
    for tid, idx in term2idx.items():
        graph[idx] = list(get_real_parents(tid))

    # 2) Propagate terms_per_entryid_hot → list index đã mở rộng
    labels_propagated = []
    for term_indices in terms_per_entryid_hot:
        expanded = set()
        stack = list(term_indices)
        while stack:
            t = stack.pop()
            if t in expanded:
                continue
            expanded.add(t)
            stack.extend(graph[t])
        labels_propagated.append(sorted(expanded))  # sort để ổn định

    return graph, labels_propagated

def is_ancestor(graph, ancestor, descendant, visited=None):
    """
    Kiểm tra xem 'ancestor' có phải là cha (hoặc ông, cụ...) của 'descendant' không
    theo đồ thị is_a.

    Args:
        graph (dict): dict term_id -> list parent_ids
        ancestor (str): term_id của ancestor
        descendant (str): term_id của descendant
        visited (set): tập các node đã thăm, dùng để tránh vòng lặp

    Returns:
        bool: True nếu ancestor là cha của descendant
    """
    if visited is None:
        visited = set()

    if descendant not in graph:
        return False

    # Tránh vòng lặp
    if descendant in visited:
        return False
    visited.add(descendant)

    parents = graph[descendant]
    if ancestor in parents:
        return True

    # Đệ quy kiểm tra các parent
    for p in parents:
        if is_ancestor(graph, ancestor, p, visited):
            return True

    return False


def build_vocab(terms, get_ancestors):
    """
       Xây dựng vocab chứa toàn bộ terms và ancestors của chúng.

       Returns:
           vocab: list term_id
           vocab_index: dict: term_id -> index
       """
    vocab = set()
    for t in terms:
        tid = t["id"]
        vocab.add(tid)
        vocab |= get_ancestors(tid)
    vocab = sorted(list(vocab))
    vocab_index = {v: i for i, v in enumerate(vocab)}
    return vocab, vocab_index


def build_term_embedding(term_id, vocab_index, get_ancestors):
    """
     Một vector nhị phân: term + ancestors được đánh dấu bằng 1.

     Returns:
         numpy array
     """
    vec = np.zeros(len(vocab_index), dtype=float)
    vec[vocab_index[term_id]] = 1.0
    for anc in get_ancestors(term_id):
        vec[vocab_index[anc]] = 1.0
    return vec


def build_embedding_matrix(terms, vocab_index, get_ancestors):
    """
       Tạo embedding matrix cho toàn bộ terms.

       Returns:
           numpy.ndarray shape (num_terms × vocab_size)
       """
    matrix = []
    for t in terms:
        emb = build_term_embedding(t["id"], vocab_index, get_ancestors)
        matrix.append(emb)
    return np.array(matrix)


def info_split(header):
    """
       Parse header FASTA và lấy EntryID, PE, SV.

       Example header:
           >sp|P31946|1433B_HUMAN Protein ... PE=1 SV=2

       Returns:
           entry_id (str)
           pe (int)
           sv (int)
       """
    parts = header.split("|")
    entry_id = parts[1] if len(parts) > 1 else None

    ox_match = re.search(r"OX=(\d+)", header)
    pe_match = re.search(r"PE=(\d+)", header)
    sv_match = re.search(r"SV=(\d+)", header)

    ox = ox_match.group(1) if ox_match else "0"
    pe = int(pe_match.group(1)) if pe_match else 0
    sv = int(sv_match.group(1)) if sv_match else 0

    return entry_id, ox, pe, sv


def build_info_matrix(fasta_file):
    """
        Đọc file FASTA và lấy EntryID, PE, SV cho mỗi protein.

        Returns:
            DataFrame: ["EntryID", "PE", "SV"]
        """
    data = []
    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                entry_id, ox, pe, sv = info_split(line)
                data.append((ox, pe, sv))

    return np.array(data)

def build_ox_test(fasta_file):
    """
    Đọc file FASTA và trả về danh sách OX (organism) theo thứ tự protein.
    
    Args:
        fasta_file (str): đường dẫn file FASTA
    
    Returns:
        list: danh sách OX (str)
    """
    ox = []

    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Lấy organism từ dòng header
                parts = line[1:].split()  # bỏ '>'
                organism = parts[1] if len(parts) > 1 else ""
                ox.append(organism)
    
    return ox

def build_mask(y, n_terms, graph):
    """
    Trả về mask_fn(i) → mask của sample i.
    
    y: list[list[int]]
    graph: list[list[int]], graph[t] = parents of t
    """

    # 1) Tìm root terms
    no_parent_mask = np.array([len(graph[t]) == 0 for t in range(n_terms)], dtype=bool)

    # 2) parent → children
    parent_to_children = defaultdict(list)
    for child, parents in enumerate(graph):
        for p in parents:
            parent_to_children[p].append(child)

    # 3) Tạo hàm lazy mask
    def mask_fn(i):
        """Trả về mask (bool array, shape = (n_terms,)) cho sample i."""
        terms = y[i]  # các term dương của sample i
        mask_i = np.zeros(n_terms, dtype=bool)

        # Root luôn True
        mask_i[no_parent_mask] = True

        # Với mỗi parent nó có → bật children
        for p in terms:
            for child in parent_to_children.get(p, []):
                mask_i[child] = True

        return mask_i

    return mask_fn

def build_proteins(fasta_file, protein_path, split_character):
    """
    Đọc file FASTA và trích tên protein (từ dòng header) theo đúng thứ tự.
    
    Returns:
        list: danh sách tên protein (str)
    """
    proteins = []

    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Lấy tên protein, bỏ dấu '>' và khoảng trắng đầu/cuối
                line = line[1:]
                protein_name = line.split(split_character)[protein_path]

                proteins.append(protein_name)
    
    return proteins

def get_vocab():
    protein_vocab = build_proteins(train_seq_file, protein_path=1, split_character='|')
    terms_vocab, terms, weights2idx = parse_obo(obo_file, ia_file)

    info = build_info_matrix(train_seq_file)
    info = info[:, 0]
    ox_vocab = list(set(info))

    df = pd.read_csv(terms_file, sep="\t")
    terms_per_entryid, term2idx, weights, terms_vocab = get_terms_per_entryid(df, terms_vocab, weights2idx, top_k=top_k, top_k_type=top_k_type)
    graph, y = build_is_a_graph(terms, term2idx, terms_per_entryid)

    return protein_vocab, terms_vocab, info, ox_vocab, graph, y, weights

def load_train():
    protein_vocab, terms_vocab, info, ox_vocab, graph, y, weights = get_vocab()
    amino_axit = get_emb_amino(train_emb_npy)

    ox, _ = one_hot(info, ox_vocab)

    x = (amino_axit, ox)

    mask = build_mask(y, len(terms_vocab), graph)

    return *x, y, mask, protein_vocab, terms_vocab, ox_vocab, weights

def load_test():
    protein_vocab, terms_vocab, _, ox_vocab, graph, _, _ = get_vocab()
    test_vocab = build_proteins(test_seq_unknown, protein_path=0, split_character=' ')
    ox = build_ox_test(test_seq_unknown)
    amino_axit = get_emb_amino(test_emb_npy)

    ox_str = [str(x) for x in ox]

    ox_onehot, _ = one_hot(ox_str, ox_vocab)

    X = (
        test_vocab,
        amino_axit,
        ox_onehot
    )
    return *X, protein_vocab, terms_vocab, ox_vocab, graph

def split_test_proteins(test_fasta_file, output_dir, terms_file, test_npy=None):
    """
    Chia test file thành protein đã biết và protein mới.
    Ghi các protein đã biết ra file TSV với xác suất 1.0 cho mỗi term.
    Ghi các protein chưa biết ra FASTA.
    Nếu có test_npy, lọc các hàng chỉ giữ protein unknown.
    
    Args:
        test_fasta_file (str)
        output_dir (str)
        terms_file (str)
        test_npy (str, optional): đường dẫn file numpy array, shape=(num_proteins, ...)
    
    Returns:
        known_file, unknown_fasta_file, unknown_npy (nếu test_npy không None)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    known_file = os.path.join(output_dir, "submit_already_known.tsv")
    unknown_fasta_file = os.path.join(output_dir, "test_unknown.fasta")

    # 1. đọc terms: protein -> list term có prob = 1.0
    df_terms = pd.read_csv(terms_file, sep=r"\s+", header=None, names=["Protein", "Term", "Prob"])
    df_prob1 = df_terms[df_terms["Prob"] == 1.0]
    protein2terms = df_prob1.groupby("Protein")["Term"].apply(list).to_dict()

    known_lines = []
    unknown_lines = []
    unknown_proteins_idx = []  # lưu index của protein unknown nếu có test_npy

    current_protein = None
    current_ox = ""
    current_seq = []
    protein_idx = 0  # index theo thứ tự protein trong FASTA

    with open(test_fasta_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing test FASTA"):
            line = line.strip()
            if line.startswith(">"):
                # xử lý protein trước đó
                if current_protein is not None:
                    if current_protein in protein2terms:
                        for t in protein2terms[current_protein]:
                            known_lines.append(f"{current_protein}\t{t}\t1.0")
                    else:
                        unknown_lines.append(f">{current_protein} {current_ox}")
                        unknown_lines.extend(current_seq)
                        unknown_proteins_idx.append(protein_idx)

                    protein_idx += 1

                # reset protein mới
                parts = line[1:].split()
                current_protein = parts[0]
                current_ox = parts[1] if len(parts) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line)

    # xử lý protein cuối cùng
    if current_protein is not None:
        if current_protein in protein2terms:
            for t in protein2terms[current_protein]:
                known_lines.append(f"{current_protein}\t{t}\t1.0")
        else:
            unknown_lines.append(f">{current_protein} {current_ox}")
            unknown_lines.extend(current_seq)
            unknown_proteins_idx.append(protein_idx)

    # ghi file known và unknown
    with open(known_file, "w", encoding="utf-8") as f:
        f.write("\n".join(known_lines))

    with open(unknown_fasta_file, "w", encoding="utf-8") as f:
        f.write("\n".join(unknown_lines))

    print(f"File known: {known_file} ({len(known_lines)} lines)")
    print(f"File unknown FASTA: {unknown_fasta_file} ({len(unknown_lines)//2} proteins)")

    # nếu có test_npy, lọc các hàng tương ứng protein unknown
    unknown_npy = None
    if test_npy is not None:
        arr = np.load(test_npy)
        unknown_npy = arr[unknown_proteins_idx]
        unknown_npy_file = os.path.join(output_dir, "test_unknown.npy")
        np.save(unknown_npy_file, unknown_npy)
        print(f"Unknown numpy saved: {unknown_npy_file} ({unknown_npy.shape[0]} proteins)")
        return known_file, unknown_fasta_file, unknown_npy_file

    return known_file, unknown_fasta_file

if __name__ == "__main__":
    #build_mask()

    #load_data()
    # X, terms_name = load_test()
    # proteins, ox, amino_acids = X
    split_test_proteins(test_seq_file, test_dir, presubmit_file)
    # In 5 protein đầu tiên
    # for i in range(min(5, len(proteins))):
    #     print(f"Protein {i + 1}:")
    #     print("ID:", proteins[i])
    #     print("OX (one-hot):", ox[i])
    #     print("Amino acids (first 30):", amino_acids[i][:30])
    #     print("Total amino acids:", len(amino_acids[i]))
    #     print("-" * 50)
