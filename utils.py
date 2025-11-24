import numpy as np



def one_hot(arr, vocab):
    # 2. Tạo mapping string -> index
    # ox2idx = {x: i for i, x in enumerate(vocab)}
    #
    # # 3. Tạo one-hot matrix
    # one_hot = np.zeros((len(arr), len(vocab)), dtype=int)
    # for i, x in enumerate(arr):
    #     j = ox2idx[x]
    #     one_hot[i, j] = 1
    #
    # return one_hot, ox2idx
    if "UNK" not in vocab:
        vocab = list(vocab) + ["UNK"]

        # 2. Tạo mapping string -> index
    ox2idx = {x: i for i, x in enumerate(vocab)}

    # 3. Chuyển arr sang string để khớp vocab
    arr_str = [str(x) for x in arr]

    # 4. Tạo one-hot matrix
    one_hot_mat = np.zeros((len(arr_str), len(vocab)), dtype=int)
    for i, x in enumerate(arr_str):
        j = ox2idx.get(x, ox2idx["UNK"])  # nếu không có trong vocab -> UNK
        one_hot_mat[i, j] = 1

    return one_hot_mat, ox2idx