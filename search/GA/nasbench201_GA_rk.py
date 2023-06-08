from score import *
import copy
import random
import os
import numpy as np
from .GA_rk_abstract import abstract_GA, chromosome

class GA(abstract_GA):
    def __init__(self, **kwargs):
        self.acc_type = kwargs["acc_type"]
        del kwargs["acc_type"]
        super().__init__(**kwargs)
        self.MAXN_CONNECTION = 6
        self.MAXN_OPERATION = 5
        self.NAS_201_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

    def init_population(self):
        self.population = [chromosome("".join([(bin(1<<random.randint(0, self.MAXN_OPERATION-1))[2:]).zfill(self.MAXN_OPERATION) for j in range(self.MAXN_CONNECTION)])) for i in range(self.MAXN_POPULATION)]

    def evaluate(self):
        for i in range(len(self.population)):
            network, uid, acc = gene2net(self.population[i].gene, self.NAS_201_ops, self.searchspace, self.acc_type, self.args.valid, self.args)
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
        p = random.randint(0, self.MAXN_CONNECTION-1)
        gene_sect_len = self.MAXN_OPERATION
        gene = chrom.gene
        genelist = [gene[i:i+gene_sect_len] for i in range(0, len(gene), gene_sect_len)]
        idx = genelist[p].find('1')
        newop = random.choice([i for i in range(idx)] + [i for i in range(idx + 1, self.MAXN_OPERATION)])
        genelist[p] = (bin(1<<newop)[2:]).zfill(self.MAXN_OPERATION)
        chrom.gene = "".join(genelist)
        return chrom

    def crossover(self, p0, p1):
        gene_sect_len = self.MAXN_OPERATION
        genelist0 = [p0.gene[i:i+gene_sect_len] for i in range(0, len(p0.gene), gene_sect_len)]
        genelist1 = [p1.gene[i:i+gene_sect_len] for i in range(0, len(p1.gene), gene_sect_len)]
        
        l0 = random.randint(1, self.MAXN_CONNECTION)
        r0 = random.choice([i for i in range(l0)] + [i for i in range(l0 + 1, self.MAXN_CONNECTION)])
        if l0>r0: l0,r0 = r0, l0
        l1 = random.randint(1, self.MAXN_CONNECTION)
        r1 = random.choice([i for i in range(l1)] + [i for i in range(l1 + 1, self.MAXN_CONNECTION)])
        if l1>r1: l1,r1 = r1, l1
        newgenelist0 = genelist0[:l1] + genelist1[l1:r1] + genelist0[r1:]
        newgenelist1 = genelist1[:l0] + genelist0[l0:r0] + genelist1[r0:]

        return chromosome("".join(newgenelist0)), chromosome("".join(newgenelist1))

    def uid2net(self, uid):
        network = self.searchspace.get_network(uid, self.args)
        acc = self.searchspace.get_final_accuracy(uid, self.acc_type, self.args.valid)
        return network, uid, acc

def gene2sect(gene, ops):
    gene_sect_len = len(ops)
    return [ops[gene[i:i+gene_sect_len].find("1")] for i in range(0, len(gene), gene_sect_len)]

def gene2net(gene, ops, searchspace, acc_type, valid, args):
    gene_sect = gene2sect(gene, ops)
    arch = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*gene_sect)
    uid = searchspace.query_index_by_arch(arch)
    network = searchspace.get_network(uid, args)
    acc = searchspace.get_final_accuracy(uid, acc_type, valid)
    return network, uid, acc
