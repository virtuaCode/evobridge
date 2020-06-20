#!/usr/bin/env python

import unittest

from ..lib.graycode import grayToStdbin


class TestGrayCodeMethods(unittest.TestCase):

    def test_grayToStdbin(self):

        self.assertEqual(list(map(int, grayToStdbin(
            bytearray([0, 1, 2, 3, 4, 5, 6, 30]))[0:8])), list(map(int, bytearray([0, 1, 3, 2, 7, 6, 4, 20]))))


if __name__ == '__main__':
    unittest.main()
