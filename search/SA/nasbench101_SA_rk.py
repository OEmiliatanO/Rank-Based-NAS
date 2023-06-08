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
        self.NAS_101_ops = ['conv_1x1', 'conv_3x3', 'maxpool']
        self.Encoder = encoder.get_encoder("nasbench101")

    def neighbor(self, arch):
        random.shuffle(arch[1])
        pos = random.sample([*range(0,5)], random.randint(0,5))
        for p in pos:
            arch[2][p] = (arch[2][p] + random.randint(0,2)) % 3
        for i in range(len(arch)):
            arch[i] = tuple(arch[i])
        return arch

    def rand_arch_generate(self):
        return self.Encoder.get_rand_code()

    def list2arch(self, l):
        return self.searchspace.get_spec_by_arch(*self.Encoder.parse_code(list(l[0]), list(l[1]), list(l[2])))
