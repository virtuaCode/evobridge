import random


def local_search(eval, mutate, accept, species, max_iter=1000):
    t = 0
    A = [species(t)]
    F = [eval(A[t])]

    while t < max_iter:
        B = mutate(A[t])
        Bf = eval(B)
        t += 1
        if accept(F[t-1], Bf, t):
            A.append(B)
            F.append(Bf)
        else:
            A.append(A[-1])
            F.append(F[-1])

    return (A, F)


def threshold_accept(temp=100, damping=0.99):
    return lambda Af, Bf, t: Bf > Af or abs(Af - Bf) <= damping ** t * temp


def onebit_mutate(genotype):
    assert type(genotype) is bytearray
    child = bytearray(len(genotype))

    for i, b in enumerate(genotype):
        child[i] = b

    index = random.randint(0, 8*len(genotype)-1)
    pos = index % 8
    child[index//8] ^= 1 << pos

    return child


def propbit_mutate(genotype):
    assert type(genotype) is bytearray

    mutrate = 0.125/len(genotype)
    child = bytearray(len(genotype))

    for i, b in enumerate(genotype):
        child[i] = b
        mask = 0
        for k in range(8):
            mask = mask << 1
            if random.random() < mutrate:
                mask ^= 1

        child[i] ^= mask

    return child
