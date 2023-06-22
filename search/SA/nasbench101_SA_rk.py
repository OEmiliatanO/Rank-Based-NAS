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
        arch = [list(x) for x in arch]
        
        # randomize the branch
        for i in range(len(arch[1])):
            arch[1][i] = random.randint(0, 1)
        
        # pruning the branch
        matrix = np.zeros([7,7])
        previous = -1
        for x,v in enumerate(arch[0]):
            if v==0:
                continue
            if previous==-1:
                matrix[0][x+1] = 1
            else:
                matrix[previous+1][x+1] = 1
            previous = x
        matrix[previous+1][-1] = 1
        map_backbone = matrix[np.triu_indices_from(matrix,k=1)]
        combined_m = 2*map_backbone + np.array(arch[1])

        zeros = np.count_nonzero(combined_m == 0)
        if zeros<12:
            del_ones_index = random.sample(list(*np.where(combined_m==1)),k=12-zeros)
            for x in del_ones_index:
                arch[1][x] = 0

        # randomize the operations
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
