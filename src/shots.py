"""Estimator tables for block Pauli coordinates from the outcome frequencies of the covering settings.

Each of the 9 covering settings (b=2) is measured with its share of the shots. A Pauli string is estimated from
every setting whose bases match it on its support and the estimates are averaged (weight-1 strings appear in 3
settings, weight-2 strings in 1).
"""
import numpy as np

import qemlib as L
from blockgeo_qem_fixed import _parity_signs

_TABLES = {}


def tables(n, b=2):
    key = (n, b)
    if key not in _TABLES:
        settings = L.measurement_settings(n, b)
        blocks = L.contiguous_blocks(n, b)
        strings = L.local_pauli_strings(b, include_identity=False)
        D, M = 2 ** n, len(blocks) * len(strings)
        signs = np.zeros((D, M))
        cover = np.zeros((len(settings), M))
        for k, blk in enumerate(blocks):
            for j, st in enumerate(strings):
                col = k * len(strings) + j
                support = {blk[b - 1 - pos]: c for pos, c in enumerate(st) if c != "I"}
                signs[:, col] = _parity_signs(n, tuple(sorted(support)))
                for si, s in enumerate(settings):
                    cover[si, col] = float(all(s[q] == c for q, c in support.items()))
        zidx = [i for i, s in enumerate(settings) if all(c == "Z" for c in s)][0]
        _TABLES[key] = (signs, cover / cover.sum(0, keepdims=True), zidx, len(blocks), len(strings))
    return _TABLES[key]
