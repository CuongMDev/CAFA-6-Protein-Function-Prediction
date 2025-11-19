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


def get_terms_per_entryid(df):
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
    return terms_per_entryid

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
    entry_id = parts[1] if len(parts) > 0 else None

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
                    nums = [ord(c) - ord('A') for c in amino_axit]
                    data.append(nums)
                parts = line.split("|")
                entry_id = parts[1] if len(parts) > 1 else None
                seq_lines = []
            else:
                seq_lines.append(line)

        # if entry_id is not None:
        #     amino_axit = "".join(seq_lines)
        #     data.append({"Protein": entry_id, "Amino_axit": amino_axit})

    return data

def load_data():
    protein = build_amino_axit_matrix(train_seq_file)
    terms_name, terms = parse_obo(obo_file)

    info = build_info_matrix(train_seq_file)
    ox, ox_vocab = one_hot(info[:, 0])

    df = pd.read_csv(terms_file, sep="\t")

    x = (protein, ox)
    y = get_terms_per_entryid(df)
    ox2idx = {x: i for i, x in enumerate(terms_name)}
    y_idx = [[ox2idx[label] for label in sublist] for sublist in y]

    return *x, y_idx, protein, terms_name, ox_vocab

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

    # graph = build_is_a_graph(terms)
    # get_ancestors = build_ancestor_lookup(graph)
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
    
    load_data()
