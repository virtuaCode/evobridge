import random


def create_threshold_accept(temp, damping, minimize=True):
    def accept(Af, Bf, t):
        threshold = damping ** t * temp
        return Bf < Af or abs(Af - Bf) <= threshold, threshold

    return accept


def create_onebit_mutate():
    def mutate(genotype):
        assert type(genotype) is bytearray
        child = bytearray(len(genotype))

        for i, b in enumerate(genotype):
            child[i] = b

        index = random.randint(0, 8*len(genotype)-1)
        pos = index % 8
        child[index//8] ^= 1 << pos

        return child
    return mutate


def create_propbit_mutate():
    def mutate(genotype):
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
    return mutate
