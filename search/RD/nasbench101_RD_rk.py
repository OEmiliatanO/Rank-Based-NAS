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

    def search(self, rk_fn_names):
        codebase = self.Encoder.get_nrand_code(self.args.n_samples)
        indices = np.array([self.searchspace.query_index_by_arch(self.searchspace.get_spec_by_arch(*self.Encoder.parse_code(c[0], c[1], c[2]))) for c in codebase])

        bestuid, taus, maxacc, rk_maxacc, times = self.ranking2(indices, rk_fn_names)

        return bestuid, taus, maxacc, rk_maxacc, times
