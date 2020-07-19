from .graycode import grayToStdbin, stdbinToGray
import numpy as np


def fromGenotype(nodes: bytearray, members: bytearray):
    nodes_ = grayToStdbin(nodes)
    members_ = grayToStdbin(members)

    n = int(len(nodes_)/2)
    l = len(members_)

    m = int((1/2) * ((8*l + 4 * n**2 - 4*n + 1)**(1/2) + 1))

    mmat = np.full((n, m), np.nan)

    j = 0
    for i in range(n):
        r = m - n
        mmat[i, 0:i+r+1] = np.mod(members_[j+i:r+j+1+2*i], 3)
        j = j + r + i

    return np.array(nodes_, dtype=float).reshape(-1, 2), mmat.transpose()


def toGenotype(nodes, members):
    nodes_ = stdbinToGray(bytearray(map(int, nodes.reshape(-1))))
    members = members.transpose()
    members_ = stdbinToGray(bytearray(map(int, members[~np.isnan(members)])))

    return nodes_, members_
