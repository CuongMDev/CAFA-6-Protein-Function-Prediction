import numpy as np
import pandas as pd
from config import *
from functools import lru_cache
import re
import matplotlib.pyplot as plt

def parse_obo(path):
    """
        Đếm số lượng protein được annotate bởi mỗi term.

        Args:
            df (DataFrame): ["EntryID", "term"]
            terms_random (list[dict]): danh sách term cần xét

        Returns:
            list[int]: số protein tương ứng với mỗi term
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

    return terms


def get_ids_numpy(terms):
    """
       Trả về array chứa ID của tất cả terms.

       Args:
           terms (list[dict])

       Returns:
           np.ndarray: mảng chứa các term id
       """
    return np.array([t["id"] for t in terms])


def create_entryid_terrm(df, terms_random):
    """
       Tạo ma trận nhị phân kích thước: (num_protein × num_terms_random)

       Args:
           df (DataFrame): chứa EntryID và term
           terms_random (list[dict]): danh sách term cần xét

       Returns:
           entryidterms: list cặp (EntryID, term)
           entryid: danh sách các protein unique
           matrix: numpy matrix 0/1
       """
    terms_random_ids = [t['id'] for t in terms_random]
    filtered = df[df["term"].isin(terms_random_ids)]

    entryidterms = filtered[["EntryID", "term"]].values
    entryid = filtered["EntryID"].unique()

    entryid_term_dict = {}
    for eid, term in entryidterms:
        entryid_term_dict.setdefault(eid, set()).add(term)

    matrix = np.zeros((len(entryid), len(terms_random_ids)), dtype=int)

    for i, eid in enumerate(entryid):
        terms_in_protein = entryid_term_dict.get(eid, set())
        for j, term in enumerate(terms_random_ids):
            if term in terms_in_protein:
                matrix[i, j] = 1

    return entryidterms, entryid, matrix


def term_protein_counts(df, terms_random):
    """
    Trả về số lượng protein mà mỗi term xuất hiện.

    Args:
        df: DataFrame có cột ["EntryID", "term"]
        terms_random: list dict, mỗi dict có key 'id' của term

    Returns:
        dict: term_id -> số protein có term đó
    """
    # Lấy danh sách term muốn xét
    terms_random_ids = [t['id'] for t in terms_random]

    # Lọc DataFrame chỉ giữ các term trong terms_random_ids
    filtered = df[df["term"].isin(terms_random_ids)]

    # Tạo dict: term -> set protein
    term_protein_dict = {}
    for eid, term in filtered[["EntryID", "term"]].values:
        term_protein_dict.setdefault(term, set()).add(eid)

    # Chuyển set protein thành số lượng
    term_counts = [len(proteins) for term, proteins in term_protein_dict.items()]

    return term_counts


def build_is_a_graph(terms):
    """
        Tạo đồ thị kế thừa (is_a graph) giữa các GO terms.

        Returns:
            dict: term_id -> danh sách các parent_id
        """
    graph = {}
    for t in terms:
        tid = t["id"]
        parents = []
        if "is_a" in t:
            for x in t["is_a"]:
                parent_id = x.split(" ! ")[0]
                parents.append(parent_id)
        graph[tid] = parents
    return graph

def build_ancestor_lookup(graph):
    """
        Tạo hàm get_ancestors(term) trả về tập tổ tiên.

        Args:
            graph (dict): term -> parent list

        Returns:
            function: lookup ancestor sử dụng lru_cache để tăng tốc
        """
    @lru_cache(None)
    def get_ancestors(term):
        parents = graph.get(term, [])
        ancestors = set(parents)
        for p in parents:
            ancestors |= get_ancestors(p)
        return ancestors
    return get_ancestors


def build_vocab(terms_random, get_ancestors):
    """
       Xây dựng vocab chứa toàn bộ terms_random và ancestors của chúng.

       Returns:
           vocab: list term_id
           vocab_index: dict: term_id -> index
       """
    vocab = set()
    for t in terms_random:
        tid = t["id"]
        vocab.add(tid)
        vocab |= get_ancestors(tid)
    vocab = sorted(list(vocab))
    vocab_index = {v:i for i,v in enumerate(vocab)}
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


def build_embedding_matrix(terms_random, vocab_index, get_ancestors):
    """
       Tạo embedding matrix cho toàn bộ terms_random.

       Returns:
           numpy.ndarray shape (num_terms × vocab_size)
       """
    matrix = []
    for t in terms_random:
        emb = build_term_embedding(t["id"], vocab_index, get_ancestors)
        matrix.append(emb)
    return np.array(matrix)

def pe_sv(header):
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
    entry_id = parts[1] if len(parts) > 0 else None

    pe_match = re.search(r"PE=(\d+)", header)
    sv_match = re.search(r"SV=(\d+)", header)

    pe = int(pe_match.group(1)) if pe_match else 0
    sv = int(sv_match.group(1)) if sv_match else 0

    return entry_id, pe, sv

def build_pe_sv_matrix(fasta_file):
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
                entry_id, pe, sv = pe_sv(line)
                data.append((entry_id, pe, sv))

    df = pd.DataFrame(data, columns=["EntryID", "PE", "SV"])
    return df

def build_amino_axit_matrix(fasta_file):
    """
       Đọc file FASTA và trích chuỗi amino acid cho mỗi protein.

       Returns:
           DataFrame: ["Protein", "Amino_axit"]
       """
    data = []
    entry_id = None
    seq_line = []

    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if entry_id is not None:
                    amino_axit = "".join(seq_lines)
                    data.append({"Protein": entry_id, "Amino_axit": amino_axit})
                parts = line.split("|")
                entry_id = parts[1] if len(parts) > 1 else None
                seq_lines = []
            else:
                seq_lines.append(line)

        # if entry_id is not None:
        #     amino_axit = "".join(seq_lines)
        #     data.append({"Protein": entry_id, "Amino_axit": amino_axit})

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    terms = parse_obo(obo_file)
    terms_random = terms[:1000]

    ids_np = get_ids_numpy(terms)
    print(ids_np)

    df = pd.read_csv(terms_file, sep="\t")
    entry_ids = df["EntryID"].unique()

    entry_ids_np = np.array(entry_ids)
    print(len(entry_ids_np))

    entryid_term, entryid, matrix = create_entryid_terrm(df, terms_random)
    print(matrix.shape)
    print(matrix)

    term_counts = term_protein_counts(df, terms)  # mỗi protein có bao nhiêu term
    #print(term_counts)
    # Đếm số protein có cùng số lượng term
    # unique_counts, protein_counts = np.unique(term_counts, return_counts=True)
    #
    # plt.figure(figsize=(10, 6))
    # plt.bar(unique_counts, protein_counts, color='skyblue')
    # plt.xlabel("Number of terms per protein")
    # plt.ylabel("Number of proteins")
    # plt.title("Distribution of number of terms per protein")
    # plt.xticks(unique_counts)  # hiển thị đầy đủ các số
    # plt.show()

    lst = np.array(term_counts)  # nếu là list, chuyển sang numpy array

    # Lấy index của 2000 phần tử lớn nhất
    top_indices = np.argsort(lst)[-1500:]  # sắp xếp tăng dần → lấy 2000 cuối cùng
    top_indices = top_indices[::-1]
    print(lst[top_indices])

    # lấy term cuủa bọn top này
    # chuyển axitamin -> numpy one-hot kiểu A->1 B->2 -> numpy 2d
    # chuyển pe sv
    # sort mảng đầu theo thứ tự term nhiều kiểu cột đầu n toàn đc tick

    # graph = build_is_a_graph(terms)
    # get_ancestors = build_ancestor_lookup(graph)
    #
    # vocab, vocab_index = build_vocab(terms_random, get_ancestors)
    #
    # print(len(vocab))
    #
    # matrix2 = build_embedding_matrix(terms_random, vocab_index, get_ancestors)
    #
    # print(matrix2.shape)
    # print(matrix2)
    #
    # df_pe_sv = build_pe_sv_matrix(train_seq_file)
    # print(df_pe_sv.head())
    #
    # protein = build_amino_axit_matrix(train_seq_file)
    # print(len(protein))
    # print(protein.head())
