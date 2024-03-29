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
        self.NATS_SSS_ops = [8, 16, 24, 32, 40, 48, 56, 64]
        self.Encoder = encoder.get_encoder("natsbenchsss")

    def neighbor(self, arch):
        new_arch = arch
        pos = random.sample([*range(5)], random.randint(0, 5))
        for p in pos:
            new_arch[p] = (new_arch[p] + random.randint(0, 7)) % 8
        return new_arch
        
    def rand_arch_generate(self):
        return list(self.Encoder.get_rand_code())

    def list2arch(self, l):
        return "{}:{}:{}:{}:{}".format(*map(lambda op: self.NATS_SSS_ops[op], l))
