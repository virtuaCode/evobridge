#!/usr/bin/env python
import pstats
from pstats import SortKey
p = pstats.Stats('profile')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
