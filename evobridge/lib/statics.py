# %%
import numpy as np

NODES = np.array([[
    3, 1,
    2, 0,
    1, 1,
    0, 0,
    0, 1,
    3, 0
]], dtype=np.float).reshape((-1, 2))

MEMBERS = np.array([[
    2, 0,
    1, 0,
    2, 1,
    4, 2,
    3, 2,
    3, 1,
    # 3, 4,
    1, 5,
    5, 0
]], dtype=np.int).reshape((-1, 2))

SUPPORTS = np.array([[3, 1, 1], [4, 1, 1]])

LOADS = np.array([[5, 0, -1]])

EXAMPLE_CONFIG = {
    "Nodes": NODES,
    "Members": MEMBERS,
    "Supports": SUPPORTS,
    "Loads": LOADS
}


# %%
def solve(config=EXAMPLE_CONFIG, node_weight=1):
    Nodes = config["Nodes"]
    Members = config["Members"]
    Supports = config["Supports"]
    Loads = config["Loads"]

    # Add node weight forces

    node_loads = np.array([x for x in np.arange(
        len(Nodes)) if x not in Supports[:, 0]]).reshape(-1, 1)

    full = np.full((node_loads.shape[0], 2), [0, -node_weight])

    Loads = np.vstack((Loads, np.hstack((node_loads, full))))

    Ns = Nodes.shape[0]
    Ms = Members.shape[0]
    Ss = sum(sum(Supports[:, 1:]))

    Amat = np.zeros((Ms+Ss, Ms+Ss))
    bmat = np.zeros((Ms+Ss, 1))

    iS = 0

    for i, node in enumerate(Nodes):
        xn, yn = node
        (K, L) = (Members == i).nonzero()

        for l in range(K.shape[0]):
            if L[l] == 0:
                xi = Nodes[Members[K[l], 1], 0]
                yi = Nodes[Members[K[l], 1], 1]
            else:
                xi = Nodes[Members[K[l], 0], 0]
                yi = Nodes[Members[K[l], 0], 1]

            si = np.hypot(xi-xn, yi-yn)

            Amat[2*i, K[l]] = (xi-xn)/si
            Amat[2*i+1, K[l]] = (yi-yn)/si

        (K2,) = (Supports[:, 0] == i).ravel().nonzero()

        if len(K2) > 0:
            if Supports[K2, 1] > 0:
                iS += 1
                Amat[2*i, Ms+iS-1] = Supports[K2, 1]
            if Supports[K2, 2] > 0:
                iS += 1
                Amat[2*i+1, Ms+iS-1] = Supports[K2, 2]

        (K3,) = (Loads[:, 0] == i).nonzero()

        for l in K3:
            bmat[2*i, 0] -= Loads[l][1]
            bmat[2*i+1, 0] -= Loads[l][2]

    detA = np.linalg.det(Amat)

    if np.abs(detA) < 1e-12:
        return (0, None, None)

    x = np.linalg.inv(Amat).dot(bmat)

    MFmat = x[0:Ms]
    RFmat = x[Ms:]

    return (detA, MFmat, RFmat)

# print(Nodes)
# print(Members)


# %%
#(detA, MFmat, RFmat) = solve(config=config)


# %%
