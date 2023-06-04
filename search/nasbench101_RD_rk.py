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
        taus = {"rk":[], "ni": [], "naswot": [], "logsynflow": []}
        cnt = 0
        codebase = self.Encoder.get_nrand_code(self.args.n_samples)
        indices = np.array([self.searchspace.query_index_by_arch(self.searchspace.get_spec_by_arch(*self.Encoder.parse_code(c[0], c[1], c[2]))) for c in codebase])
        overhead = time.time() - overhead_st

        niuid, naswotuid, logsynuid, bestrk_uid, rk_tau, ni_tau, naswot_tau, logsyn_tau, maxacc, rk_maxacc, ni_time, naswot_time, logsynflow_time, rk_time = self.ranking(indices, [1,1,1], cnt)

        return niuid, naswotuid, logsynuid, bestrk_uid, rk_tau, ni_tau, naswot_tau, logsyn_tau, maxacc, rk_maxacc, ni_time + overhead, naswot_time + overhead, logsynflow_time + overhead, rk_time + overhead
