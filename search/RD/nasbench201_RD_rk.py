import random
import numpy as np
import torch
import sys
import time
from score import *
from .RD_rk_abstract import abstract_RD

class RD(abstract_RD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def search(self):
        overhead_st = time.time()
        cnt = 0
        codebase = self.Encoder.get_nrand_code(self.args.n_samples)
        indices = np.array([self.searchspace.get_index_by_code(self.Encoder.parse_code(c)) for c in codebase])
        overhead = time.time() - overhead_st

        bestuid, taus, maxacc, rk_maxacc, times = self.ranking(indices, [1,1,1], cnt)
        times = tuple([x+overhead for x in times])

        return bestuid, taus, maxacc, rk_maxacc, times
