from .graycode import grayToStdbin, stdbinToGray
import numpy as np


def fromGenotype(nodes, members, materials):
    nodes_ = nodes
    members_ = members
    materials_ = materials

    l = len(members_)

    m = int(0.5 * (1 + (l*8 + 1)**0.5))

    mmem = np.full((m, m), False, dtype=np.bool)
    mmem[np.triu_indices(m, 1)] = members_

    mmat = np.full((m, m), False, dtype=np.bool)
    mmat[np.triu_indices(m, 1)] = materials_

    return nodes_.reshape(-1, 2), mmem, mmat


def toGenotype(nodes, members, materials):
    nodes_ = nodes.reshape(-1)
    members_ = members[np.triu_indices(members.shape[0], 1)]
    materials_ = materials[np.triu_indices(members.shape[0], 1)]

    return nodes_, members_, materials_
