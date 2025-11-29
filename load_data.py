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

def get_terms_per_entryid(protein_vocab, df, terms_name, weight, top_k, top_k_type):
    """
    Tạo danh sách term-index cho mỗi protein trong protein_vocab.
    Top-k theo aspect được đưa lên đầu.
    """
    if len(top_k) != 3 or len(top_k_type) != 3:
        raise ValueError("top_k và top_k_type phải có độ dài 3")

    k1, k2, k3 = top_k

    # -----------------------------
    # 2) Tần suất & aspect
    # -----------------------------
    term_counter = Counter(df["term"].values)
    aspect_dict = dict(zip(df["term"], df["aspect"]))

    # -----------------------------
    # 3) Lấy top-k theo aspect
    # -----------------------------
    top_terms = {a: [] for a in top_k_type}
    k_dict = {top_k_type[0]: k1, top_k_type[1]: k2, top_k_type[2]: k3}

    for term, _ in term_counter.most_common():
        if term not in terms_name:
            continue
        asp = aspect_dict.get(term, None)
        if asp in top_terms and len(top_terms[asp]) < k_dict[asp]:
            top_terms[asp].append(term)

    # -----------------------------
    # 4) Tạo filtered_terms_list
    # -----------------------------
    topk_ordered = []
    for a in top_k_type:
        topk_ordered.extend(top_terms[a])

    remaining_terms = [t for t in terms_name if t not in topk_ordered]
    filtered_terms_list = topk_ordered + remaining_terms

    # Aspect list theo filtered_terms_list
    filtered_aspects_list = [aspect_dict.get(t, None) for t in filtered_terms_list]

    most_common_terms = set(filtered_terms_list)

    # -----------------------------
    # 5) Map EntryID → set(term)
    # -----------------------------
    entryid_term_dict = {}
    for eid, term in df[["EntryID", "term"]].values:
        if term in most_common_terms:
            entryid_term_dict.setdefault(eid, set()).add(term)

    # -----------------------------
    # 6) Map sang index theo protein_vocab
    # -----------------------------
    term2idx = {t: i for i, t in enumerate(filtered_terms_list)}

    terms_per_entryid = []
    for protein in protein_vocab:
        # Lấy đúng terms của protein, nếu không có → []
        terms = entryid_term_dict.get(protein, set())
        idx_list = [
            term2idx[t] for t in terms 
            if t in term2idx and term2idx[t] < NUM_CLASSES
        ]
        terms_per_entryid.append(idx_list)


    # -----------------------------
    # 7) Weight list theo thứ tự filtered_terms_list
    # -----------------------------
    term_weights_list = [weight[t] for t in filtered_terms_list]

    return (
        terms_per_entryid,        # theo đúng protein_vocab
        term2idx,
        term_weights_list,
        filtered_terms_list,
        filtered_aspects_list
    )


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
    Build full DAG và graph nối parent chỉ gồm node < NUM_CLASSES.

    Returns:
        graph_full: list[list[int]] toàn bộ parent
        graph_masked: list[list[int]] chỉ node < NUM_CLASSES và parent < NUM_CLASSES
        labels_propagated: list[list[int]] nhãn mở rộng < NUM_CLASSES
    """
    # --- Build temp parent mapping ---
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
        for p in temp_parents[tid]:
            if p in term2idx:
                real_parents.add(term2idx[p])
            else:
                real_parents |= get_real_parents(p)
        memo[tid] = real_parents
        return real_parents

    n_terms = len(term2idx)
    graph_full = [[] for _ in range(n_terms)]
    graph_masked = [[] for _ in range(NUM_CLASSES)]

    for tid, idx in term2idx.items():
        parents_idx = list(get_real_parents(tid))
        graph_full[idx] = parents_idx
        # Chỉ giữ parent < NUM_CLASSES cho graph_masked
        if idx < NUM_CLASSES:
            graph_masked[idx] = [p for p in parents_idx if p < NUM_CLASSES]

    # --- Propagate labels per entry, chỉ < NUM_CLASSES ---
    labels_propagated = []
    for term_indices in terms_per_entryid_hot:
        expanded = set()
        stack = list(term_indices)
        while stack:
            t = stack.pop()
            if t in expanded:
                continue
            if t < NUM_CLASSES:
                expanded.add(t)
            stack.extend(graph_full[t])
        labels_propagated.append(sorted(expanded))

    return graph_full, graph_masked, labels_propagated

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


def build_train_proteins(fasta_file):
    """
        Đọc file FASTA và lấy EntryID, PE, SV cho mỗi protein.

        Returns:
            DataFrame: ["EntryID", "PE", "SV"]
        """
    data = []
    protein = []
    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                entry_id, ox, pe, sv = info_split(line)
                data.append((ox, pe, sv))
                protein.append(entry_id)

    return protein, np.array(data)

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

def build_test_proteins(fasta_file):
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
                protein_name = line.split(' ')[0]

                proteins.append(protein_name)
    
    return proteins

def read_submission_tsv(file, protein_vocab, terms_vocab):
    """
    Đọc file TSV submission dạng Protein\tTerm và trả về dict:
        protein_idx -> {term_idx: 1}

    Args:
        file (str): đường dẫn file TSV
        protein_vocab (list): danh sách protein, index = protein_idx
        terms_vocab (list): danh sách term, index = term_idx

    Returns:
        dict: {protein_idx: {term_idx: 1, ...}}
    """
    # tạo lookup dict để tra cứu nhanh
    protein2idx = {p: i for i, p in enumerate(protein_vocab)}
    term2idx = {t: i for i, t in enumerate(terms_vocab)}

    protein_to_terms_dict = {}

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            protein, term = parts[:2]

            # map sang index
            if protein not in protein2idx or term not in term2idx:
                continue
            p_idx = protein2idx[protein]
            t_idx = term2idx[term]

            if p_idx not in protein_to_terms_dict:
                protein_to_terms_dict[p_idx] = {}
            protein_to_terms_dict[p_idx][t_idx] = 1  # gán giá trị 1 cho term

    return protein_to_terms_dict

def get_vocab():
    protein_vocab, info = build_train_proteins(train_seq_file)
    info = info[:, 0]
    terms_vocab, terms, weights2idx = parse_obo(obo_file, ia_file)

    ox_vocab = list(set(info))

    df = pd.read_csv(terms_file, sep="\t")
    terms_per_entryid, term2idx, weights, terms_vocab, aspects_list = get_terms_per_entryid(protein_vocab, df, terms_vocab, weights2idx, top_k=top_k, top_k_type=top_k_type)
    graph, graph_masked, y = build_is_a_graph(terms, term2idx, terms_per_entryid)

    return protein_vocab, terms_vocab, info, ox_vocab, graph, graph_masked, y, weights, aspects_list

def filter_train(terms_per_entryid, aspects_list, required_aspects, min_match=2):
    """
    Lọc những entry mà có ít nhất `min_match` aspect trong required_aspects.

    Args:
        terms_per_entryid: list[list[int]]
            Danh sách nhãn dạng index theo filtered_terms_list.
        aspects_list: list[str]
            filtered_aspects_list, cùng thứ tự terms_vocab.
        required_aspects: set hoặc list
            Ví dụ: {'C','F','P'}
        min_match: int
            Số aspect tối thiểu phải có trong entry (mặc định 2)

    Returns:
        filtered_terms_per_entryid: list[list[int]]
            Chỉ giữ entry có đủ các aspect.
        valid_indices: list[int]
            Index của các entry trong batch gốc.
    """

    return terms_per_entryid, range(len(terms_per_entryid))
    
    # required_aspects = set(required_aspects)

    # filtered = []
    # valid_indices = []

    # for idx, term_indices in enumerate(terms_per_entryid):
    #     # Lấy aspect thực tế của từng term trong entry
    #     entry_aspects = {aspects_list[i] for i in term_indices}

    #     # Check có ít nhất min_match trong required_aspects
    #     if len(entry_aspects & required_aspects) >= min_match:
    #         filtered.append(term_indices)
    #         valid_indices.append(idx)

    # return filtered, valid_indices

def load_train():
    protein_vocab, terms_vocab, info, ox_vocab, _, graph_masked, y, weights, aspects_list = get_vocab()
    y, indices = filter_train(y, aspects_list, top_k_type)
    amino_axit = np.concatenate((get_emb_amino(t5_train_emb_npy), get_emb_amino(esm_train_emb_npy)), axis=-1)[indices]

    ox, _ = one_hot(info[indices], ox_vocab)

    x = (amino_axit, ox)

    mask = build_mask(y, NUM_CLASSES, graph_masked)

    return *x, y, mask, protein_vocab, terms_vocab, ox_vocab, weights[:NUM_CLASSES], graph_masked

def load_test():
    protein_vocab, terms_vocab, _, ox_vocab, graph, graph_masked, _, _, _ = get_vocab()
    test_vocab = build_test_proteins(test_seq_file)
    ox = build_ox_test(test_seq_file)
    amino_axit = np.concatenate((get_emb_amino(t5_test_emb_npy), get_emb_amino(esm_test_emb_npy)), axis=-1)

    ox_str = [str(x) for x in ox]

    ox_onehot, _ = one_hot(ox_str, ox_vocab)

    protein_to_terms_dict = read_submission_tsv(presubmit_file, protein_vocab, terms_vocab)

    X = (
        test_vocab,
        amino_axit,
        ox_onehot
    )
    return *X, protein_vocab, terms_vocab, ox_vocab, graph, graph_masked, protein_to_terms_dict

def export_known_proteins(test_fasta_file, output_dir, terms_file):
    """
    Xuất file TSV chỉ gồm protein đã biết với các term có Prob == 1.0.
    Không xuất unknown proteins, không gán prob=1.0.
    
    Args:
        test_fasta_file (str)
        output_dir (str)
        terms_file (str)
    
    Returns:
        known_file (str)
    """
    import os
    import pandas as pd
    from tqdm import tqdm

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    known_file = os.path.join(output_dir, "known_proteins.tsv")

    # đọc terms: chỉ giữ các term có Prob == 1.0
    df_terms = pd.read_csv(terms_file, sep=r"\s+", header=None, names=["Protein", "Term", "Prob"])
    df_terms = df_terms[df_terms["Prob"] == 1.0]  # lọc xác suất bằng 1.0
    protein2terms = df_terms.groupby("Protein")["Term"].apply(list).to_dict()

    known_lines = []

    with open(test_fasta_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing test FASTA"):
            if line.startswith(">"):
                protein = line[1:].split()[0]
                if protein in protein2terms:
                    for term in protein2terms[protein]:
                        known_lines.append(f"{protein}\t{term}")

    # ghi file
    with open(known_file, "w", encoding="utf-8") as f:
        f.write("\n".join(known_lines))

    print(f"File known proteins saved: {known_file} ({len(known_lines)} lines)")
    return known_file

if __name__ == "__main__":
    #build_mask()

    #load_data()
    # X, terms_name = load_test()
    # proteins, ox, amino_acids = X
    export_known_proteins(test_seq_file, test_dir, presubmit_file)
    # In 5 protein đầu tiên
    # for i in range(min(5, len(proteins))):
    #     print(f"Protein {i + 1}:")
    #     print("ID:", proteins[i])
    #     print("OX (one-hot):", ox[i])
    #     print("Amino acids (first 30):", amino_acids[i][:30])
    #     print("Total amino acids:", len(amino_acids[i]))
    #     print("-" * 50)
