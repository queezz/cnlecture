import os
import sys

sys.path.insert(0, os.path.abspath("../"))

from cnlecture import functions as fc


def test_primitive_roots():
    fc.primitive_roots(5)
