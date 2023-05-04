from score import *
import copy
import random
import os
import numpy as np

class chromosome():
    def __init__(self, gene = None, fitness = None, acc = None, uid = None):
        """ fitness: (naswot, ntk, ni, logsynflow, tot) """
        self.gene = gene
        self.fitness = fitness
        self.acc = acc
        self.uid = uid

class GA():
    def __init__(self, MAXN_CONNECTION, MAXN_OPERATION, searchspace, train_loader, device, stds, means, acc_type, args):
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
        self.best_chrom = {"ni": chromosome(), "naswot": chromosome(), "logsynflow": chromosome(), "ninaswot": chromosome()}
        self.candiate = {"ninaswot":set(), "ntk":set(), "logsynflow":set(), "naswot":set(), "ni":set(), "tot":[]}
        self.NAS_201_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.NATS_SSS_ops = [8, 16, 24, 32, 40, 48, 56, 64]

    def init_population(self):
        if self.args.nasspace == 'natsbenchSSS':
            self.population = [chromosome(random.sample(range(0,len(self.NATS_SSS_ops)), k=5)) for i in range(self.MAXN_POPULATION)]
        elif self.args.nasspace == 'nasbench201':
            self.population = [chromosome("".join([(bin(1<<random.randint(0, self.MAXN_OPERATION-1))[2:]).zfill(self.MAXN_OPERATION) for j in range(self.MAXN_CONNECTION)])) for i in range(self.MAXN_POPULATION)]

    def evaluate(self):
        for i in range(len(self.population)):
            if self.args.nasspace == 'natsbenchSSS':
                network, uid, acc = gene2net(self.population[i].gene, self.NATS_SSS_ops, self.searchspace, self.acc_type, self.args.valid, self.args)
            elif self.args.nasspace == 'nasbench201':
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
        if self.args == "natsbenchSSS":
            p = random.randint(0, 7)
            chrom.gene[p] = random.randint(0, 7)
        elif self.args == "nasbench201":
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
        totrk = dict(zip([chrom.uid for chrom in cand], [0 for i in range(N)]))
        rk_naswot     = sorted(cand, key = lambda x: x.fitness[0], reverse = True)
        rk_ni         = sorted(cand, key = lambda x: x.fitness[1], reverse = True)
        rk_logsynflow = sorted(cand, key = lambda x: x.fitness[2], reverse = True)
        for rk, chrom in enumerate(rk_ni):
            totrk[chrom.uid] += rk
        
        for rk, chrom in enumerate(rk_naswot):
            totrk[chrom.uid] += rk

        bestrk = np.inf
        for rk, chrom in enumerate(rk_logsynflow):
            totrk[chrom.uid] += rk

        chosen = [*map(lambda x: x[0], sorted(totrk.items(), key = lambda x: x[1])[:2])]
        res = [*filter(lambda x: x.uid == chosen[0] or x.uid == chosen[1], cand)]
        return res

    def crossover(self, p0, p1):
        if self.args.nasspace == 'natsbenchSSS':
            l0 = random.randint(1, 8)
            r0 = random.choice([i for i in range(l0)] + [i for i in range(l0 + 1, 8)])
            if l0>r0: l0,r0 = r0, l0
            l1 = random.randint(1, 8)
            r1 = random.choice([i for i in range(l1)] + [i for i in range(l1 + 1, 8)])
            if l1>r1: l1,r1 = r1, l1
            newgene0 = p0.gene[:l1] + p1.gene[l1:r1] + p0.gene[r1:]
            newgene1 = p1.gene[:l0] + p0.gene[l0:r0] + p1.gene[r0:]
            return chromosome(newgene0), chromosome(newgene1)
        elif self.args.nasspace == 'nasbench201':
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
        assert False, f"doesn't support such nasspace: {self.args.nasspace}"
        return None

    def add_candiates_by(self, idx, metrice):
        self.population.sort(key = lambda this: this.fitness[idx], reverse = True)
        for met in metrice:
            self.candiate[met].add(self.population[0].uid)
            self.candiate[met].add(self.population[1].uid)

    def find_best(self):
        self.init_population()
        self.evaluate()
        for i in range(self.MAXN_ITERATION):
            offsprings = []
            elder = []
            while len(offsprings) + len(elder) < self.MAXN_POPULATION:
                p = self.select_2chrom_fromN()
                
                if random.randint(0,99) < self.PROB_CROSS*100:
                    offspring0, offspring1 = self.crossover(p[0], p[1])
                else:
                    elder.append(p[0])
                    elder.append(p[1])
                    offspring0, offspring1 = None, None
 
                if random.randint(0,99) < self.PROB_MUTATION*100:
                    offspring0 = self.mutation(offspring0)
                    offspring1 = self.mutation(offspring1)
                
                if offspring0 and offspring1:
                    offsprings.append(offspring0)
                    offsprings.append(offspring1)
            
            self.population = offsprings
            self.evaluate()
            offsprings = self.population
            self.population = elder + offsprings
            
            self.add_candiates_by(0, ["naswot", "ni", "logsynflow"])
            self.add_candiates_by(1, ["naswot", "ni", "logsynflow"])
            self.add_candiates_by(2, ["naswot", "ni", "logsynflow"])

            self.population = elder + offsprings
        
        rk_naswot     = sorted(list(self.candiate["naswot"]), key = lambda this: self.DICT[this][0], reverse = True)
        rk_ni         = sorted(list(self.candiate["ni"]), key = lambda this: self.DICT[this][1], reverse = True)
        rk_logsynflow = sorted(list(self.candiate["logsynflow"]), key = lambda this: self.DICT[this][2], reverse = True)

        totrk = dict(zip([uid for uid in self.candiate["naswot"]], [0 for i in range(len(rk_naswot))]))
    
        for rk, id in enumerate(rk_ni):
            totrk[id] += rk
        
        for rk, id in enumerate(rk_naswot):
            totrk[id] += rk

        bestrk = np.inf
        for rk, id in enumerate(rk_logsynflow):
            totrk[id] += rk

            if bestrk > totrk[id]:
                bestrk = totrk[id]
                bestrk_uid = id
               
        network, uid, acc = uid2net(bestrk_uid, self.searchspace, self.acc_type, self.args.valid, self.args)
        
        if self.args.verbose:
            return self.DICT[uid], acc, uid, best_rank, self.best_chrom["naswot"].acc, self.best_chrom["ni"].acc, self.best_chrom["logsynflow"].acc, self.best_chrom["ninaswot"].acc
        else:
            return self.DICT[uid], acc, uid, bestrk

def gene2sect(gene, ops):
    gene_sect_len = len(ops)
    return [ops[gene[i:i+gene_sect_len].find("1")] for i in range(0, len(gene), gene_sect_len)]

def gene2net(gene, ops, searchspace, acc_type, valid, args):
    if args.nasspace == "natsbenchSSS":
        arch = "{}:{}:{}:{}:{}".format(*map(lambda x: ops[x], gene))
    elif args.nasspace == "nasbench201":
        gene_sect = gene2sect(gene, ops)
        arch = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*gene_sect)
    uid = searchspace.query_index_by_arch(arch)
    network = searchspace.get_network(uid, args)
    acc = searchspace.get_final_accuracy(uid, acc_type, valid)
    return network, uid, acc

def uid2net(uid, searchspace, acc_type, valid, args):
    network = searchspace.get_network(uid, args)
    acc = searchspace.get_final_accuracy(uid, acc_type, valid)
    return network, uid, acc
