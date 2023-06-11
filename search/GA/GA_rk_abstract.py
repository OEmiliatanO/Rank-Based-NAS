from score import *
import copy
import random
import os
import numpy as np

class chromosome():
    def __init__(self, gene = None, fitness = None, acc = None, uid = None):
        self.gene = gene
        self.fitness = fitness
        self.acc = acc
        self.uid = uid

class abstract_GA():
    def __init__(self, searchspace, train_loader, device, args):
        self.MAXN_POPULATION = args.maxn_pop
        self.MAXN_ITERATION = args.maxn_iter
        self.PROB_MUTATION = args.prob_mut
        self.PROB_CROSS = args.prob_cr
        self.DICT = {}
        self.population = []
        self.searchspace = searchspace
        self.train_loader = train_loader
        self.args = args
        self.device = device
        self.best_chrom = {"ni": chromosome(), "naswot": chromosome(), "logsynflow": chromosome()}
        self.candiate = {"logsynflow":set(), "naswot":set(), "ni":set()}

    def init_population(self):
        #TODO
        pass

    def mutation(self, chrom):
        #TODO
        pass

    def evaulate(self):
        #TODO
        pass

    def crossover(self):
        #TODO
        pass

    def uid2net(self, uid):
        #TODO
        pass

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

        for rk, chrom in enumerate(rk_logsynflow):
            totrk[chrom.uid] += rk

        chosen = [*map(lambda x: x[0], sorted(totrk.items(), key = lambda x: x[1])[:2])]
        res = [*filter(lambda x: x.uid == chosen[0] or x.uid == chosen[1], cand)]
        return res

    def add_candiates_by(self, idx, metrice):
        self.population.sort(key = lambda this: this.fitness[idx], reverse = True)
        for met in metrice:
            self.candiate[met].add(self.population[0].uid)
            self.candiate[met].add(self.population[1].uid)

    def search(self):
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
               
        network, uid, acc = self.uid2net(bestrk_uid)
        
        if self.args.verbose:
            return self.DICT[uid], acc, uid, best_rank, self.best_chrom["naswot"].acc, self.best_chrom["ni"].acc, self.best_chrom["logsynflow"].acc
        else:
            return self.DICT[uid], acc, uid, bestrk

