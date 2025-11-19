import numpy as np

def one_hot(arr, vocab=None):
    if vocab is None:
        # 1. Lấy danh sách unique theo thứ tự xuất hiện
        unique_arr = []
        seen = set()
        for x in arr:
            if x not in seen:
                seen.add(x)
                unique_arr.append(x)
    else:
        unique_arr = vocab

    # 2. Tạo mapping string -> index
    ox2idx = {x: i for i, x in enumerate(unique_arr)}

    # 3. Tạo one-hot matrix
    one_hot = np.zeros((len(arr), len(unique_arr)), dtype=int)
    for i, x in enumerate(arr):
        j = ox2idx[x]
        one_hot[i, j] = 1

    return one_hot, ox2idx