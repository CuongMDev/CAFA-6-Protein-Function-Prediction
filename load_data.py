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
    graph = [] * len(term2idx)
    for t in terms:
        tid = term2idx[t["id"]]
        parents = []
        if "is_a" in t:
            for x in t["is_a"]:
                parent_id = x.split(" ! ")[0]
                parents.append(term2idx[parent_id])
        graph[tid] = parents
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

    ox = int(ox_match.group(1)) if pe_match else 0
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
                    nums = [ord(c) - ord('A') for c in seq_str if 'A' <= c <= 'Z']
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
            nums = [ord(c) - ord('A') for c in seq_str if 'A' <= c <= 'Z']
            protein.append(entry_id)
            amino_axit.append(nums)

    return protein, amino_axit

def build_protein_ox_aminoaxit_test(fasta_file):
    protein = []
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
            protein_id = parts[0] if len(parts) > 0 else ""
            organism = parts[1] if len(parts) > 1 else ""

            protein.append(protein_id)
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

    ox_onehot, ox2idx = one_hot(ox)

    return protein, ox_onehot, amino_axit


def build_mask(y, n_terms, graph):
    """
    Vectorized mask creation using NumPy.

    Args:
        y: list of list of term indices per protein
        n_terms: total number of terms
        graph: dict term_idx -> list of parent_idx

    Returns:
        mask: np.array of shape (n_samples, n_terms), dtype=bool
    """
    n_samples = len(y)

    # 1. Convert y to dense array (n_samples, n_terms)
    y_array = np.zeros((n_samples, n_terms), dtype=bool)
    for i, terms in enumerate(y):
        y_array[i, terms] = True

    # 2. Build parent adjacency matrix (n_terms, n_terms)
    # parent_adj[t, p] = True if p is parent of t
    parent_adj = np.zeros((n_terms, n_terms), dtype=bool)
    for t, parents in graph.items():
        if parents:
            parent_adj[t, parents] = True

    # 3. Compute mask
    # Terms with no parent → train always
    no_parent_mask = parent_adj.sum(axis=1) == 0  # shape (n_terms,)

    # Terms with parents → at least one parent present
    # y_array: (n_samples, n_terms)
    # parent_adj: (n_terms, n_terms)
    # Broadcasting: (n_samples,1,n_terms) & (1,n_terms,n_terms) -> (n_samples,n_terms,n_terms)
    parent_presence = (y_array[:, np.newaxis, :] & parent_adj[np.newaxis, :, :]).any(axis=2)

    # Combine with no-parent terms
    mask = parent_presence | no_parent_mask[np.newaxis, :]

    return mask


def load_data():
    protein = build_protein_amino_axit_matrix(train_seq_file)
    terms_name, terms = parse_obo(obo_file)

    info = build_info_matrix(train_seq_file)
    ox, ox_vocab = one_hot(info[:, 0])

    df = pd.read_csv(terms_file, sep="\t")

    x = (protein, ox)
    y, term2idx = get_terms_per_entryid_hot(df, terms_name)

    graph = build_is_a_graph(terms, term2idx)

    mask = build_mask(y, len(terms_name), graph)

    return *x, y, mask, protein, terms_name, ox_vocab

def load_test():
    protein, ox, amino_axit = build_protein_ox_aminoaxit_test(test_seq_file)


    X = (
        protein,
        ox,
        amino_axit
    )
    terms_name = parse_obo(obo_file)

    return X, terms_name

if __name__ == "__main__":
    # terms = parse_obo(obo_file)
    # terms = terms[:1000]

    # ids_np = get_ids_numpy(terms)

    # df = pd.read_csv(terms_file, sep="\t")
    # entry_ids = df["EntryID"].unique()

    # entry_ids_np = np.array(entry_ids)

    # entryid_term, entryid, matrix = create_entryid_terrm(df, terms)
    # print(matrix.shape)
    # print(matrix)

    # term_counts = term_protein_counts(df, terms)  # mỗi protein có bao nhiêu term

    # lst = np.array(term_counts)  # nếu là list, chuyển sang numpy array

    # # Lấy index của 2000 phần tử lớn nhất
    # top_indices = np.argsort(lst)[-1500:]  # sắp xếp tăng dần → lấy 2000 cuối cùng
    # top_indices = top_indices[::-1]
    # print(lst[top_indices])

    # lấy term cuủa bọn top này
    # chuyển axitamin -> numpy one-hot kiểu A->1 B->2 -> numpy 2d
    # chuyển pe sv
    # sort mảng đầu theo thứ tự term nhiều kiểu cột đầu n toàn đc tick

    #get_ancestors = build_ancestor_lookup(graph)
    #
    # vocab, vocab_index = build_vocab(terms, get_ancestors)
    #
    # print(len(vocab))
    #
    # matrix2 = build_embedding_matrix(terms, vocab_index, get_ancestors)
    #
    # print(matrix2.shape)
    # print(matrix2)
    #
    # df_pe_sv = build_pe_sv_matrix(train_seq_file)
    # print(df_pe_sv.head())
    #

    #load_data()
    X, terms_name = load_test()
    proteins, ox, amino_acids = X

    # In 5 protein đầu tiên
    for i in range(min(5, len(proteins))):
        print(f"Protein {i + 1}:")
        print("ID:", proteins[i])
        print("OX (one-hot):", ox[i])
        print("Amino acids (first 30):", amino_acids[i][:30])
        print("Total amino acids:", len(amino_acids[i]))
        print("-" * 50)
