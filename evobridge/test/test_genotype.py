#!/usr/bin/env python

import unittest

from ..lib.genotype import fromGenotype, toGenotype
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
        nodes = np.array([[1, 1], [43, 32], [4, 3], [9, 3]])
        members = np.array(
            [1, 2, 3, 4,
             1, 1, 1, 1,
             1, 1, 6, 1,
             1, 1, 1, 1,
             0, 1, 1, 1,
             0, 0, 1, 1,
             0, 0, 0, 1,
             0, 0, 0, 0], dtype=float).reshape(-1, 4)
        members[members == 0] = np.nan

        a1, a2 = fromGenotype(*toGenotype(nodes, members))

        members[np.isnan(members)] = 0
        a2[np.isnan(a2)] = 0

        self.assertTrue((a1 == nodes).all())
        self.assertTrue((a2 == members).all())


if __name__ == '__main__':
    unittest.main()
