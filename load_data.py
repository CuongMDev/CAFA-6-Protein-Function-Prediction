from collections import defaultdict
import numpy as np
import pandas as pd
from config import *
from functools import lru_cache
import re
import matplotlib.pyplot as plt
from utils import one_hot


def parse_obo(path):
    """
        Đếm số lượng protein được annotate bởi mỗi term.

        Args:
            terms (list[dict]): danh sách term cần xét

        Returns:
            list[int]: số protein tương ứng với mỗi term
        """
    terms = []
    terms_name = []
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
                    terms_name.append(term["id"])
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

    return terms_name, terms


def get_ids_numpy(terms):
    """
       Trả về array chứa ID của tất cả terms.

       Args:
           terms (list[dict])

       Returns:
           np.ndarray: mảng chứa các term id
       """
    return np.array([t["id"] for t in terms])


def get_terms_per_entryid_hot(df, terms_name):
    """
    Trả về list các list term mà mỗi EntryID có.

    Args:
        df (DataFrame): chứa cột 'EntryID' và 'term'

    Returns:
        terms_per_entryid: list các list term của mỗi EntryID
    """
    entryids = df["EntryID"].unique()

    entryid_term_dict = {}
    for eid, term in df[["EntryID", "term"]].values:
        entryid_term_dict.setdefault(eid, set()).add(term)

    terms_per_entryid = [list(entryid_term_dict.get(eid, set())) for eid in entryids]

    term2idx = {x: i for i, x in enumerate(terms_name)}
    terms_per_entryid_hot = [[term2idx[label] for label in sublist] for sublist in terms_per_entryid]

    return terms_per_entryid_hot, term2idx


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


def build_is_a_graph(terms, term2idx):
    """
        Tạo đồ thị kế thừa (is_a graph) giữa các GO terms.

        Returns:
            dict: term_id -> danh sách các parent_id
        """
    graph = [[] for _ in range(len(term2idx))]
    for t in terms:
        tid = term2idx[t["id"]]
        if "is_a" in t:
            for x in t["is_a"]:
                parent_id = x.split(" ! ")[0]
                graph[tid].append(term2idx[parent_id])
    return graph


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


def build_protein_amino_axit_matrix(fasta_file):
    """
    Đọc file FASTA và trích chuỗi amino acid cho mỗi protein.

    Returns:
        tuple: (amino_acids, proteins)
    """
    amino_axit = []
    protein = []
    entry_id = None
    seq_lines = []

    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # bỏ dòng rỗng
                continue

            if line.startswith(">"):
                # Append protein trước đó
                if seq_lines:
                    seq_str = "".join(seq_lines)
                    nums = [ord(c) - ord('A') + 1 for c in seq_str if 'A' <= c <= 'Z']
                    protein.append(entry_id)
                    amino_axit.append(nums)

                # Lấy entry_id mới
                parts = line.split("|")
                entry_id = parts[1] if len(parts) > 1 else f"unknown_{len(protein)}"
                seq_lines = []
            else:
                seq_lines.append(line.upper())  # chuẩn hóa chữ hoa

        # Append protein cuối cùng
        if seq_lines:
            seq_str = "".join(seq_lines)
            nums = [ord(c) - ord('A') + 1 for c in seq_str if 'A' <= c <= 'Z']
            protein.append(entry_id)
            amino_axit.append(nums)

    return protein, amino_axit

def build_ox_aminoaxit_test(fasta_file):
    amino_axit = []
    ox = []

    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Kiểm tra dòng bắt đầu bằng '>' (header)
        if line.startswith('>'):
            # Tách protein ID và OX
            parts = line[1:].split()  # Bỏ dấu '>' và tách
            organism = parts[1] if len(parts) > 1 else ""

            ox.append(organism)

            # Đọc chuỗi amino acid (có thể nhiều dòng)
            i += 1
            sequence = ""
            while i < len(lines) and not lines[i].startswith('>'):
                sequence += lines[i].strip()
                i += 1

            # Chuyển đổi amino acid thành số (A=1, B=2, ..., Z=26)
            amino_numbers = [ord(aa) - ord('A') + 1 for aa in sequence if aa.isalpha()]
            amino_axit.append(amino_numbers)
        else:
            i += 1

    return ox, amino_axit

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

def get_vocab():
    protein_vocab, amino_axit = build_protein_amino_axit_matrix(train_seq_file)
    terms_vocab, terms = parse_obo(obo_file)

    info = build_info_matrix(train_seq_file)
    info = info[:, 0]
    ox_vocab = list(set(info))
    if "UNK" not in ox_vocab:
        ox_vocab.append("UNK")
    return protein_vocab, amino_axit, terms_vocab, terms, info, ox_vocab

def load_train():
    protein_vocab, amino_axit, terms_vocab, terms, info, ox_vocab = get_vocab()

    ox_str = [str(x) for x in info]
    ox, _ = one_hot(ox_str, ox_vocab)

    df = pd.read_csv(terms_file, sep="\t")

    x = (amino_axit, ox)
    y, term2idx = get_terms_per_entryid_hot(df, terms_vocab)

    graph = build_is_a_graph(terms, term2idx)

    mask = build_mask(y, len(terms_vocab), graph)

    return x[0], x[1], y, mask, protein_vocab, terms_vocab, ox_vocab

def load_test():
    protein_vocab, _, terms_vocab, _, _, ox_vocab = get_vocab()
    ox, amino_axit = build_ox_aminoaxit_test(test_seq_file)

    ox_str = [str(x) for x in ox]

    ox_onehot = one_hot(ox_str, ox_vocab)

    X = (
        protein_vocab,
        amino_axit,
        ox_onehot
    )
    return X[0], X[1], X[2], protein_vocab, terms_vocab, ox_vocab

if __name__ == "__main__":
    #build_mask()

    #load_data()
    # X, terms_name = load_test()
    # proteins, ox, amino_acids = X
    load_test()
    # In 5 protein đầu tiên
    # for i in range(min(5, len(proteins))):
    #     print(f"Protein {i + 1}:")
    #     print("ID:", proteins[i])
    #     print("OX (one-hot):", ox[i])
    #     print("Amino acids (first 30):", amino_acids[i][:30])
    #     print("Total amino acids:", len(amino_acids[i]))
    #     print("-" * 50)
