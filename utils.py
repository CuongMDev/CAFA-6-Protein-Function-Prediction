import numpy as np



def one_hot(arr, vocab):
    # 2. Tạo mapping string -> index
    ox2idx = {x: i for i, x in enumerate(vocab)}

    # 3. Tạo one-hot matrix
    one_hot = np.zeros((len(arr), len(vocab)), dtype=int)
    for i, x in enumerate(arr):
        j = ox2idx[x]
        one_hot[i, j] = 1

    return one_hot, ox2idx