from score import *
import copy
import random
import numpy as np

class chromosome():
    def __init__(self, gene = "", fitness = (0,0,0), acc = 0, uid = 0):
        """ fitness: (ninaswot, ntk, entropy) """
        self.gene = gene
        self.fitness = fitness
        self.acc = acc
        self.uid = uid

###
base = "/home/jasonzzz/Genetic-Based-Neural-Architecture-Search-with-Hybrid-Score-Functions/results/20230306"
###

class GA():
    def __init__(self, MAXN_CONNECTION, MAXN_OPERATION, searchspace, train_loader, device, stds, means, acc_type, args):
        #### cheat
        self.ninaswot = np.load(f"{base}/ninaswot_nasbench201_cifar10_none_0.05_1_True_128_1_1.npy")
        self.ntk      = np.load(f"{base}/ntk_nasbench201_cifar10_none_0.05_1_True_128_1_1.npy")
        self.entropy  = np.load(f"{base}/entropy_nasbench201_cifar10_none_0.05_1_True_128_1_1.npy")
        assert len(self.ninaswot) == 15625, "broken: ninaswot"
        assert len(self.ntk) == 15625, "broken: ntk"
        assert len(self.entropy) == 15625, "broken: entropy"
        ####
        self.MAXN_POPULATION = args.maxn_pop
        self.MAXN_ITERATION = args.maxn_iter
        self.PROB_MUTATION = args.prob_mut
        self.PROB_CROSS = args.prob_cr
        self.MAXN_CONNECTION = MAXN_CONNECTION
        self.MAXN_OPERATION = MAXN_OPERATION
        self.DICT = {}
        self.population = []
        self.searchspace = searchspace
        self.train_loader = train_loader
        self.args = args
        self.device = device
        self.stds = stds
        self.means = means
        self.acc_type = acc_type
        self.best_chrom = chromosome()
        self.proposition = eval(args.proposition)
        self.NAS_201_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

    def init_population(self):
        self.population = [chromosome("".join([(bin(1<<random.randint(0, self.MAXN_OPERATION-1))[2:]).zfill(self.MAXN_OPERATION) for j in range(self.MAXN_CONNECTION)])) for i in range(self.MAXN_POPULATION)]
        """
        for i in range(self.MAXN_POPULATION):
            chrom = chromosome("".join([(bin(1<<random.randint(0, self.MAXN_OPERATION-1))[2:]).zfill(self.MAXN_OPERATION) for j in range(self.MAXN_CONNECTION)]))
            #for j in range(self.MAXN_CONNECTION):
            #    op = random.randint(0, self.MAXN_OPERATION-1)
            #    chrom.gene += (bin(1<<op)[2:]).zfill(self.MAXN_OPERATION)
            self.population.append(chrom)
        """

    def evaluate(self):
        for i in range(len(self.population)):
            network, uid, acc = gene2net(self.population[i].gene, self.NAS_201_ops, self.searchspace, self.acc_type, self.args.trainval)
            self.population[i].acc = acc
            self.population[i].uid = uid
            if self.population[i].gene not in self.DICT:
                #self.population[i].fitness = self.DICT[self.population[i].gene] = \
                #(standardize(ninaswot_score(network, self.train_loader, self.device, self.stds, self.means, self.args), self.means["ninaswot"], self.stds["ninaswot"]), \
                # -standardize(ntk_score(network, self.train_loader, self.device, train_mode=self.args.trainval), self.means["ntk"], self.stds["ntk"]), \
                # standardize(entropy_score(network, self.train_loader, self.device, self.args), self.means["entropy"], self.stds["entropy"]))
                self.population[i].fitness = self.DICT[self.population[i].gene] = (self.ninaswot[self.population[i].uid], self.ntk[self.population[i].uid], 0)
            else:
                self.population[i].fitness = self.DICT[self.population[i].gene]
            # global optimum
            if sum(self.population[i].fitness) > sum(self.best_chrom.fitness) or self.best_chrom.gene == "":
                self.best_chrom.fitness = self.population[i].fitness
                self.best_chrom.acc = self.population[i].acc
                self.best_chrom.uid = self.population[i].uid
                self.best_chrom.gene = copy.deepcopy(self.population[i].gene)
            
            del network

    def mutation(self, chrom):
        p = random.randint(0, self.MAXN_CONNECTION-1)
        gene_sect_len = self.MAXN_OPERATION
        gene = chrom.gene
        genelist = [gene[i:i+gene_sect_len] for i in range(0, len(gene), gene_sect_len)]
        idx = genelist[p].find('1')
        newop = random.choice([i for i in range(idx)] + [i for i in range(idx + 1, self.MAXN_OPERATION)])
        genelist[p] = (bin(1<<newop)[2:]).zfill(self.MAXN_OPERATION)
        chrom.gene = "".join(genelist)
        return chrom
    
    def select_2chrom_fromN(self, N=3):
        cand = random.sample(self.population, N)
        maxi = smaxi = cand[0]
        for chrom in cand:
            if sum(maxi.fitness) < sum(chrom.fitness):
                smaxi = maxi
                maxi = chrom
            elif sum(smaxi.fitness) < sum(chrom.fitness):
                smaxi = chrom
        return maxi, smaxi

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
        """
        break_point = random.randint(1, self.MAXN_CONNECTION-1)
        newgenelist0 = genelist0[:break_point] + genelist1[break_point:]
        newgenelist1 = genelist1[:break_point] + genelist0[break_point:]
        """
        return chromosome("".join(newgenelist0)), chromosome("".join(newgenelist1))

    def find_best(self):
        self.init_population()
        self.evaluate()
        for i in range(self.MAXN_ITERATION):
            offsprings = []
            while len(offsprings) < self.MAXN_POPULATION:
                p = self.select_2chrom_fromN()
                
                if random.randint(0,99) < self.PROB_CROSS*100:
                    offspring0, offspring1 = self.crossover(p[0], p[1])
                else:
                    offspring0, offspring1 = p[0], p[1]
 
                if random.randint(0,99) < self.PROB_MUTATION*100:
                    offspring0 = self.mutation(offspring0)
                    offspring1 = self.mutation(offspring1)
                
                offsprings.append(offspring0)
                offsprings.append(offspring1)
            
            self.population = offsprings
            self.evaluate()
            offsprings.sort(key = lambda this: sum(this.fitness), reverse = True)
            offsprings = offsprings[:int(len(offsprings)*0.5)]
        
        #offsprings.sort(key = lambda this: this.fitness[0], reverse = True)
        #if random.randint(0,99) <= 49:
        network, uid, acc = gene2net(self.best_chrom.gene, self.NAS_201_ops, self.searchspace, self.acc_type, self.args.trainval)
        #else:
        #    network, uid, acc = gene2net(offsprings[0].gene, self.NAS_201_ops, self.searchspace, self.acc_type, self.args.trainval)
        return self.best_chrom.fitness, acc, uid

def gene2sect(gene, ops):
    gene_sect_len = len(ops)
    return [ops[gene[i:i+gene_sect_len].find("1")] for i in range(0, len(gene), gene_sect_len)]

def gene2net(gene, ops, searchspace, acc_type, trainval):
    gene_sect = gene2sect(gene, ops)
    arch = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*gene_sect)
    idx = searchspace.query_index_by_arch(arch)
    uid = searchspace[idx]
    network = searchspace.get_network(uid)
    acc = searchspace.get_final_accuracy(uid, acc_type, trainval)
    return network, uid, acc
