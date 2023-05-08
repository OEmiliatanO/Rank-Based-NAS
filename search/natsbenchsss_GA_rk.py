from score import *
import copy
import random
import os
import numpy as np
from .GA_rk_abstract import abstract_GA, chromosome

class GA(abstract_GA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.NATS_SSS_ops = [8, 16, 24, 32, 40, 48, 56, 64]

    def init_population(self):
        self.population = [chromosome(random.sample(range(0, 8), k=5)) for i in range(self.MAXN_POPULATION)]

    def evaluate(self):
        for i in range(len(self.population)):
            network, uid, acc = gene2net(self.population[i].gene, self.NATS_SSS_ops, self.searchspace, None, self.args.valid, self.args)
            #del network
            self.population[i].acc = acc
            self.population[i].uid = uid
            if self.population[i].uid not in self.DICT:
                uid = self.population[i].uid
                """ fitness: (naswot, ni, logsynflow) """
                x = [naswot_score(network, self.train_loader, self.device, self.args), \
                ni_score(network, self.train_loader, self.device, self.args), \
                logsynflow_score(network, self.train_loader, self.device)]

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
                if self.best_chrom["logsynflow"].gene == None or self.population[i].fitness[3] > self.best_chrom["logsynflow"].fitness[3]:
                    self.best_chrom["logsynflow"].fitness = self.population[i].fitness
                    self.best_chrom["logsynflow"].acc = self.population[i].acc
                    self.best_chrom["logsynflow"].uid = self.population[i].uid
                    self.best_chrom["logsynflow"].gene = copy.deepcopy(self.population[i].gene)

    def mutation(self, chrom):
        if chrom == None: return None
        p = random.randint(0, 4)
        chrom.gene[p] = random.randint(0, 7)
        return chrom
    
    def crossover(self, p0, p1):
        l0 = random.randint(1, 8)
        r0 = random.choice([i for i in range(l0)] + [i for i in range(l0 + 1, 8)])
        if l0>r0: l0,r0 = r0, l0
        l1 = random.randint(1, 8)
        r1 = random.choice([i for i in range(l1)] + [i for i in range(l1 + 1, 8)])
        if l1>r1: l1,r1 = r1, l1
        newgene0 = p0.gene[:l1] + p1.gene[l1:r1] + p0.gene[r1:]
        newgene1 = p1.gene[:l0] + p0.gene[l0:r0] + p1.gene[r0:]
        return chromosome(newgene0), chromosome(newgene1)
        return None, None

    def uid2net(self, uid):
        network = self.searchspace.get_network(uid, self.args)
        acc = self.searchspace.get_final_accuracy(uid, None, self.args.valid)
        return network, uid, acc

def gene2net(gene, ops, searchspace, acc_type, valid, args):
    arch = "{}:{}:{}:{}:{}".format(*map(lambda x: ops[x], gene))
    uid = searchspace.query_index_by_arch(arch)
    network = searchspace.get_network(uid, args)
    acc = searchspace.get_final_accuracy(uid, acc_type, valid)
    return network, uid, acc
