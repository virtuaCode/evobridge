import random
import math
import numpy as np


def create_threshold_accept(temp, damping):
    def accept(A, B, t):
        Af = np.sum(A)
        Bf = np.sum(B)

        threshold = damping ** t * temp
        return Bf < Af or abs(Af - Bf) <= threshold, threshold

    return accept


def create_simulated_annealing_accept(temp, damping):
    def accept(A, B, t):
        Af = np.sum(A)
        Bf = np.sum(B)
        threshold = math.exp(-abs(Af-Bf)/(damping**t*temp))
        if Bf < Af:
            return True, threshold

        return random.random() <= threshold, threshold

    return accept


def create_record_to_record_accept(temp, damping):
    besteF = -1

    def accept(A, B, t):
        Af = np.sum(A)
        Bf = np.sum(B)
        nonlocal besteF
        threshold = damping ** t * temp

        if Bf < besteF or besteF == -1:
            besteF = Bf
            return True, threshold

        return abs(Bf - besteF) < threshold, threshold

    return accept


def create_creep_mutate(pm, step, ps=0.05):
    def mutate(genotype):
        child = np.copy(genotype)

        for i in range(child.shape[0]):
            u = random.random()
            if u < pm:
                lower = max(0, child[i] - step)
                upper = min(255, child[i] + step)
                child[i] = round(random.uniform(lower, upper)*4)/4

            if u < ps:
                a, b = random.randint(
                    0, child.shape[0]-1), random.randint(0, child.shape[0]-1)
                child[a], child[b] = child[b], child[a]

        return child
    return mutate


def create_onebit_mutate():
    def mutate(genotype):
        child = np.copy(genotype)
        index = random.randint(0, genotype.shape[0]-1)
        child[index] = ~child[index]

        return child
    return mutate


def create_probbit_mutate():
    def mutate(genotype):
        mutrate = (6/genotype.shape[0])
        child = np.copy(genotype)

        mask = np.random.beta(2, 2, genotype.shape[0]) < mutrate

        child = child ^ mask

        return child
    return mutate


def create_tournament_select(size, k):
    def select(F):
        I = np.empty((size), dtype=np.int)
        r = len(F)
        for i in range(size):
            index = random.randint(0, r-1)
            for j in range(k-1):
                u = random.randint(0, r-1)
                F_index = F[index]
                F_u = F[u]
                if F_u < F_index:
                    index = u
            I[i] = index
        return I
    return select


def create_k_point_crossover(k1, k2, k3):
    def crossover(A, B):
        a1, a2, a3 = A
        b1, b2, b3 = B

        l1, l2, l3 = a1.shape[0], a2.shape[0], a3.shape[0]
        c1, c2, c3 = np.full(l1, 0, dtype=float), np.full(
            l2, False, dtype=np.bool), np.full(l3, False, dtype=np.bool)
        d1, d2, d3 = np.full(l1, 0, dtype=float), np.full(
            l2, False, dtype=np.bool), np.full(l3, False, dtype=np.bool)

        def _crossover(a, b, c, d, k):
            if type(a) == tuple:
                repeat = len(a)
            else:
                repeat = 1
                a = a,
                b = b,
                c = c,
                d = d,

            l = a[0].shape[0]
            j = np.random.randint(0, l, size=(k+2))
            j[0] = 0
            j[-1] = l+1
            j = np.sort(j)
            for r in range(repeat):
                for m in range(k+1):
                    if m % 2 == 0:
                        c[r][j[m]:j[m+1]] = a[r][j[m]:j[m+1]]
                        d[r][j[m]:j[m+1]] = b[r][j[m]:j[m+1]]
                    else:
                        c[r][j[m]:j[m+1]] = b[r][j[m]:j[m+1]]
                        d[r][j[m]:j[m+1]] = a[r][j[m]:j[m+1]]

        _crossover(a1, b1, c1, d1, k1)
        _crossover((a2, a3), (b2, b3), (c2, c3), (d2, d3), k2)
        #_crossover(a3, b3, c3, d3, k3)

        return (c1, c2, c3), (d1, d2, d3)
    return crossover
