import random
import numpy as np
import torch
import sys
from tqdm import trange
import time
from score import *
from .SA_rk_abstract import abstract_SA
from encoder import encoder

class SA(abstract_SA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.NAS_201_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.Encoder = encoder.get_encoder("nasbench201")

    def neighbor(self, arch):
        l1, l2, l3 = arch[:6], arch[6], arch[7:]
        pos = random.sample([*range(6)], random.randint(0, 5))
        for p in pos:
            l1[p] = (l1[p] + random.randint(0, 4)) % 5

        return l1+l2+l3

    def rand_arch_generate(self):
        return self.Encoder.get_rand_code()

    def list2arch(self, l):
        l = list(l)
        return self.searchspace.get_arch_str_by_code(self.Encoder.parse_code(l))
