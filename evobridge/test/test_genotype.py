#!/usr/bin/env python

import unittest

from ..lib.genotype import fromGenotype, toGenotype
from ..lib.functions import create_k_point_crossover
import numpy as np


class TestGrayCodeMethods(unittest.TestCase):

    def test_genotype_1(self):
        a1, a2 = fromGenotype(bytearray([0, 0, 1, 1]), bytearray([1, 1, 1]))
        b1, b2 = np.array([[0, 0], [1, 1]], dtype=float), np.array(
            [1, 1, 0, 1, 0, 0], dtype=float).reshape(-1, 2)

        a2[np.isnan(a2)] = 0

        self.assertTrue((a1 == b1).all())
        self.assertTrue((a2 == b2).all())

    def test_genotype_2(self):
        a1, a2 = fromGenotype(
            bytearray([0, 0, 1, 1, 0, 0, 1, 1]), bytearray([1]*22))
        b1, b2 = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=float), np.array(
            [1, 1, 1, 1,
             1, 1, 1, 1,
             1, 1, 1, 1,
             1, 1, 1, 1,
             0, 1, 1, 1,
             0, 0, 1, 1,
             0, 0, 0, 1,
             0, 0, 0, 0], dtype=float).reshape(-1, 4)

        a2[np.isnan(a2)] = 0

        self.assertTrue((a1 == b1).all())
        self.assertTrue((a2 == b2).all())

    def test_genotype_reverse(self):
        nodes = np.array([[1, 1], [43, 32], [4, 3], [9, 3]], dtype=float)
        members = np.tri(10, 10, -1, dtype=np.bool)
        materials = np.tri(10, 10, -1, dtype=np.bool)

        a1, a2, a3 = fromGenotype(*toGenotype(nodes, members, materials))

        self.assertTrue((a1 == nodes).all())
        self.assertTrue((a2 == members).all())
        self.assertTrue((a3 == materials).all())

    def test_crossover(self):
        nodes1 = np.arange(10)
        nodes2 = np.arange(10)[::-1]

        mem1 = np.full(10, True)
        mem2 = np.full(10, False)

        mat1 = np.full(10, True)
        mat2 = np.full(10, False)

        crossover = create_k_point_crossover(1, 1, 1)

        g1, g2 = (nodes1, mem1, mat1), (nodes2, mem2, mat2)

        g3, g4 = crossover(g1, g2)


if __name__ == '__main__':
    unittest.main()
