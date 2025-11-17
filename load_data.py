import numpy as np
import pandas as pd
from config import *
from functools import lru_cache


def parse_obo(path):
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
    return np.array([t["id"] for t in terms])


def create_entryid_terrm(df, terms_random):
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

def build_is_a_graph(terms):
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
    @lru_cache(None)
    def get_ancestors(term):
        parents = graph.get(term, [])
        ancestors = set(parents)
        for p in parents:
            ancestors |= get_ancestors(p)
        return ancestors
    return get_ancestors


def build_vocab(terms_random, get_ancestors):
    vocab = set()
    for t in terms_random:
        tid = t["id"]
        vocab.add(tid)
        vocab |= get_ancestors(tid)
    vocab = sorted(list(vocab))
    vocab_index = {v:i for i,v in enumerate(vocab)}
    return vocab, vocab_index


def build_term_embedding(term_id, vocab_index, get_ancestors):
    vec = np.zeros(len(vocab_index), dtype=float)
    vec[vocab_index[term_id]] = 1.0
    for anc in get_ancestors(term_id):
        vec[vocab_index[anc]] = 1.0
    return vec


def build_embedding_matrix(terms_random, vocab_index, get_ancestors):
    matrix = []
    for t in terms_random:
        emb = build_term_embedding(t["id"], vocab_index, get_ancestors)
        matrix.append(emb)
    return np.array(matrix)

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

    graph = build_is_a_graph(terms)
    get_ancestors = build_ancestor_lookup(graph)

    vocab, vocab_index = build_vocab(terms_random, get_ancestors)

    print(len(vocab))

    matrix2 = build_embedding_matrix(terms_random, vocab_index, get_ancestors)

    print(matrix2.shape)
    print(matrix2)