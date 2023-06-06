from score import *
import copy
import random
import os
import numpy as np
from .GA_rk_abstract import abstract_GA, chromosome
from encoder import encoder
import torch

class GA(abstract_GA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MAXN_CONNECTION = 6
        self.MAXN_OPERATION = 3
        self.NAS_101_ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        self.Encoder = encoder.get_encoder("nasbench101")
        torch.autograd.set_detect_anomaly(True)

    def init_population(self):
        s = set()
        while len(s) < self.MAXN_POPULATION:
            bb, br, op = self.Encoder.get_rand_code()
            bb, br, op = tuple(bb), tuple(br), tuple(op)
            s.add((bb,br,op))
        s = list(s)
        for i in range(len(s)):
            s[i] = list(s[i])
            s[i][0], s[i][1], s[i][2] = list(s[i][0]), list(s[i][1]), list(s[i][2])
        self.population = [chromosome(gene) for gene in s]

    def evaluate(self):
        for i in range(len(self.population)):
            network, uid, acc = gene2net(self.population[i].gene, self.Encoder, self.searchspace, None, self.args.valid, self.args)
            #del network
            self.population[i].acc = acc
            self.population[i].uid = uid
            if self.population[i].uid not in self.DICT:
                uid = self.population[i].uid
                """ fitness: (naswot, ni, synflow) """
                x = [naswot_score(network, self.train_loader, self.device, self.args), \
                ni_score(network, self.train_loader, self.device, self.args), \
                synflow_score(network, self.train_loader, self.device)]

                self.population[i].fitness = self.DICT[uid] = tuple(x)
            else:
                self.population[i].fitness = self.DICT[self.population[i].uid]
            del network
            
            # global optimum
            if self.args.verbose:
                if self.best_chrom["naswot"].gene == None or self.population[i].fitness[0] > self.best_chrom["naswot"].fitness[0]:
                    self.best_chrom["naswot"].fitness = self.population[i].fitness
                    self.best_chrom["naswot"].acc = self.population[i].acc
                    self.best_chrom["naswot"].uid = self.population[i].uid
                    self.best_chrom["naswot"].gene = copy.deepcopy(self.population[i].gene)
                if self.best_chrom["ni"].gene == None or self.population[i].fitness[1] > self.best_chrom["ni"].fitness[1]:
                    self.best_chrom["ni"].fitness = self.population[i].fitness
                    self.best_chrom["ni"].acc = self.population[i].acc
                    self.best_chrom["ni"].uid = self.population[i].uid
                    self.best_chrom["ni"].gene = copy.deepcopy(self.population[i].gene)
                if self.best_chrom["synflow"].gene == None or self.population[i].fitness[3] > self.best_chrom["synflow"].fitness[3]:
                    self.best_chrom["synflow"].fitness = self.population[i].fitness
                    self.best_chrom["synflow"].acc = self.population[i].acc
                    self.best_chrom["synflow"].uid = self.population[i].uid
                    self.best_chrom["synflow"].gene = copy.deepcopy(self.population[i].gene)

    def mutation(self, chrom):
        random.shuffle(chrom.gene[1])
        pos = random.sample([*range(0,5)], random.randint(0,5))
        for p in pos:
            chrom.gene[2][p] = (chrom.gene[2][p] + random.randint(0,2)) % self.MAXN_OPERATION
        return chrom

    def crossover(self, p0, p1):
        l0 = random.randint(0, 4)
        len = random.randint(1, 5-l0)
        r0 = l0 + len
        l1 = random.randint(0, 5-len)
        r1 = l1 + len
        p0.gene[2] = p0.gene[2][:l0] + p1.gene[2][l1:r1] + p0.gene[2][r0:]
        p1.gene[2] = p1.gene[2][:l1] + p0.gene[2][l0:r0] + p1.gene[2][r1:]
        
        l0 = random.randint(0, 20)
        len = random.randint(1, 21-l0)
        r0 = l0 + len
        l1 = random.randint(0, 20-len)
        r1 = l1 + len
        p0.gene[1] = p0.gene[1][:l0] + p1.gene[1][l1:r1] + p0.gene[1][r0:]
        p1.gene[1] = p1.gene[1][:l1] + p0.gene[1][l0:r0] + p1.gene[1][r1:]

        return p0, p1

    def uid2net(self, uid):
        network = self.searchspace.get_network(uid, self.args)
        acc = self.searchspace.get_final_accuracy(uid, self.acc_type, self.args.valid)
        return network, uid, acc

def gene2net(gene, Encoder, searchspace, acc_type, valid, args):
    m, op = Encoder.parse_code(gene[0], gene[1], gene[2])
    spec = searchspace.get_spec_by_arch(m, op)
    uid = searchspace.query_index_by_arch(spec)
    network = searchspace.get_network(uid, args)
    acc = searchspace.get_final_accuracy(uid, acc_type, valid)
    return network, uid, acc
