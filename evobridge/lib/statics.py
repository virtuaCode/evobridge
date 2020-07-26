# %%
import numpy as np
from scipy.linalg import lu

# %%


def solve(nodes, members, supports, loads):
    assert len(members) > 0
    assert len(nodes) > 0

    Ns = nodes.shape[0]
    Ms = members.shape[0]
    Ss = np.sum(supports[:, 1] + supports[:, 2])

    x = (Ms + Ss) - 2 * Ns

    if x != 0:
        return (x, None, None, None)

    Amat = np.zeros((Ms+Ss, Ms+Ss))
    bmat = np.zeros((Ms+Ss, 1))

    iS = 0

    for i, node in enumerate(nodes):
        xn, yn = node
        (K, L) = (members == i).nonzero()

        for l in range(K.shape[0]):
            if L[l] == 0:
                xi = nodes[members[K[l], 1], 0]
                yi = nodes[members[K[l], 1], 1]
            else:
                xi = nodes[members[K[l], 0], 0]
                yi = nodes[members[K[l], 0], 1]

            si = np.hypot(xi-xn, yi-yn)

            if si == 0:  # two or more nodes have same position
                return (x, 0, None, None)

            Amat[2*i, K[l]] = (xi-xn)/si
            Amat[2*i+1, K[l]] = (yi-yn)/si

        (K2,) = (supports[:, 0] == i).ravel().nonzero()

        if len(K2) > 0:
            if supports[K2, 1] > 0:
                iS += 1
                Amat[2*i, Ms+iS-1] = supports[K2, 1]
            if supports[K2, 2] > 0:
                iS += 1
                Amat[2*i+1, Ms+iS-1] = supports[K2, 2]

        (K3,) = (loads[:, 0] == i).nonzero()

        for l in K3:
            bmat[2*i, 0] -= loads[l][1]
            bmat[2*i+1, 0] -= loads[l][2]

    if False:
        p, l, u = lu(Amat)
        print("Amat", Amat)
        print(p)
        print("l", np.all(l == 0, axis=1))
        print()
        print("u", np.all(u == 0, axis=1))
        print()

    detA = np.linalg.det(Amat)

    #print("det", detA)

    if np.abs(detA) < 1e-12:
        return (x, 0, None, None)

    X = np.linalg.inv(Amat).dot(bmat)

    MFmat = X[0:Ms]
    RFmat = X[Ms:]

    return (x, detA, MFmat, RFmat)
