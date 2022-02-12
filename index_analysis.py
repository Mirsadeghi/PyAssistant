import numpy as np


# pair_id = [[1, 2], [3, 4], [5, 6], [2, 7], [8, 9]]
# input : Nx2 array pair_id
# output: Nx1 list index indicating group index of input
# group_id = getOverlapIndices(pair_id)

def getOverlapIndices(sel_pairs):
    n = sel_pairs.shape[0]
    groupIdx = np.zeros((n, )).astype('int')

    for i in range(0, n-1):
        if groupIdx[i] == 0:
            groupIdx[i] = max(groupIdx)+1
        for j in range(i+1, n):
            if groupIdx[j] == 0:
                for k in range(1, max(groupIdx)+1):
                    id_k = np.where(groupIdx==k)[0]
                    group_int = set(sel_pairs[id_k, :].reshape(-1)).intersection(set(sel_pairs[j, :]))
                    if len(group_int) > 0:
                        groupIdx[j] = k
                    else:
                        groupIdx[j] = max(groupIdx)+1
    return groupIdx        