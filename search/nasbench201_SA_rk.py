import random
import numpy as np
import torch
import sys
from tqdm import trange
import time
from score import *
from .SA_rk_abstract import abstract_SA

class SA(abstract_SA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.NAS_201_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

    def neighbor(self, arch):
        new_arch = arch
        pos = random.sample([*range(6)], random.randint(0, 5))
        for p in pos:
            new_arch[p] = (new_arch[p] + random.randint(0, 4)) % 5
        return new_arch

    def rand_arch_generate(self):
        return [random.choice([*range(5)]) for i in range(6)]

    def list2arch(self, l):
        return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*map(lambda op: self.NATS_201_ops[op], l))
