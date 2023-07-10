import random
import numpy as np
import torch
import sys
from tqdm import trange
import time
from score import *

class abstract_SA():
    def __init__(self, searchspace, train_loader, acc_type, device, args):
        self.end_T = args.end_T
        self.maxn_iter = args.maxn_iter
        self.Rt = args.Rt
        self.init_T = args.init_T
        self.maxN = args.maxN
        self.alpha = args.alpha
        self.searchspace = searchspace
        self.train_loader = train_loader
        self.device = device
        self.acc_type = acc_type
        self.args = args
        self.DICT = {}

    def neighbor(self, arch):
        #TODO
        assert False, f"implemenation isn't completed"
        pass
    
    def rand_arch_generate(self):
        #TODO
        assert False, f"implemenation isn't completed"
        pass

    def search(self):
        best_arch = now_arch = self.rand_arch_generate()
        now_sol_uid = best_sol_uid = self.searchspace.query_index_by_arch(self.list2arch(now_arch))
        ###
        history_best_arch = set()
        ###
        T = self.init_T
        while T > self.end_T:
            for i in range(self.maxn_iter):
                neighbors = set()
                while len(neighbors) < self.maxN:
                    neighbors.add(tuple(self.neighbor(list(now_arch))))
                neighbors.add(tuple(now_arch))
                ###
                for arch in history_best_arch:
                    neighbors.add(tuple(arch))
                ###
                #neighbors.add(tuple(best_arch))
                
                neighbors = list(neighbors)
                now_uid = self.searchspace.query_index_by_arch(self.list2arch(now_arch))
                indices = [self.searchspace.query_index_by_arch(self.list2arch(arch)) for arch in neighbors]
                
                # E
                rk_ni, rk_naswot, rk_logsynflow, bestrk_uid, maxacc, rk_maxacc, ind_rk = self.ranking(np.array(indices), [1,1,1])
                """
                # trans ver. 1
                dE = ind_rk[indices.index(bestrk_uid)] - ind_rk[indices.index(now_uid)]
                if np.log(random.uniform(0,1)) < dE/T:
                    pass
                else:
                    now_sol_uid = bestrk_uid
                    now_arch = neighbors[indices.index(bestrk_uid)]

                best_sol_uid = bestrk_uid
                best_arch = neighbors[indices.index(bestrk_uid)]
                """

                # trans ver. 2

                best_sol_uid = bestrk_uid
                best_arch = neighbors[indices.index(bestrk_uid)]
                history_best_arch.add(best_arch)

                bestrk = np.inf
                if bestrk_uid == now_uid:
                    now_ind = indices.index(now_uid)
                    for j in range(len(ind_rk)):
                        if bestrk > ind_rk[j] and j != now_ind:
                            bestrk = ind_rk[j]
                            bestrk_uid = indices[j]
                
                dE = ind_rk[indices.index(bestrk_uid)] - ind_rk[indices.index(now_uid)]
                #print(f"dE={ind_rk[indices.index(bestrk_uid)]}-{ind_rk[indices.index(now_uid)]}={dE}")
                #print(f"now sol acc = {self.searchspace.get_final_accuracy(now_uid, self.acc_type, self.args.valid)}")
                if dE < 0 or np.log(random.uniform(0,1)) <= -self.alpha * dE / T:
                    #print("trans.")
                    now_uid = bestrk_uid
                    now_arch = neighbors[indices.index(bestrk_uid)]
                #print(f"best sol acc = {self.searchspace.get_final_accuracy(best_sol_uid, self.acc_type, self.args.valid)}", end=', ')

            T *= self.Rt
            try:
                best_sol_uid = int(best_sol_uid)
                now_uid = int(now_uid)
            except:
                pass
        #print(f"best sol acc = {self.searchspace.get_final_accuracy(best_sol_uid, self.acc_type, self.args.valid)}", end=', ')
        #print(f"now sol acc = {self.searchspace.get_final_accuracy(now_uid, self.acc_type, self.args.valid)}")
        try:
            return best_sol_uid, self.searchspace.get_final_accuracy(best_sol_uid, self.acc_type, self.args.valid), now_uid, self.searchspace.get_final_accuracy(now_uid, self.acc_type, self.args.valid)
        except:
            print(best_sol_uid)
            print(now_uid)
    
    def ranking(self, indices, weight):
        scores = {"ni": [], "naswot": [], "logsynflow": []}
        
        for uid in indices:
            try:
                uid = int(uid)
            except:
                pass
            network = self.searchspace.get_network(uid, self.args)
            network = network.to(self.device)
            if uid not in self.DICT:
                nisc = ni_score(network, self.train_loader, self.device, self.args)
                naswotsc = naswot_score(network, self.train_loader, self.device, self.args)
                logsynflowsc = logsynflow_score(network, self.train_loader, self.device)
                scores["ni"].append(nisc)
                scores["naswot"].append(naswotsc)
                scores["logsynflow"].append(logsynflowsc)
                self.DICT[uid] = (nisc, naswotsc, logsynflowsc)
            else:
                scores["ni"].append(self.DICT[uid][0])
                scores["naswot"].append(self.DICT[uid][1])
                scores["logsynflow"].append(self.DICT[uid][2])

            del network

        totrk = dict(zip([uid for uid in indices], [0 for i in range(self.args.n_samples)]))
        
        m_ni = np.argsort(scores["ni"])
        rk_ni = indices[m_ni]
        for rk, id in enumerate(rk_ni):
            totrk[id] += (self.args.n_samples - rk) * weight[0]
        
        m_naswot = np.argsort(scores["naswot"])
        rk_naswot = indices[m_naswot]
        for rk, id in enumerate(rk_naswot):
            totrk[id] += (self.args.n_samples - rk) * weight[1]

        m_logsyn = np.argsort(scores["logsynflow"])
        rk_logsynflow = indices[m_logsyn]
        bestrk = np.inf
        for rk, id in enumerate(rk_logsynflow):
            totrk[id] += (self.args.n_samples - rk) * weight[2]
            if bestrk > totrk[id]:
                bestrk = totrk[id]
                bestrk_uid = id
        try:
            accs = [self.searchspace.get_final_accuracy(int(uid), self.acc_type, self.args.valid) for uid in indices]
        except:
            accs = [self.searchspace.get_final_accuracy(uid, self.acc_type, self.args.valid) for uid in indices]

        maxacc = np.max(accs)
        try:
            rk_maxacc = self.searchspace.get_final_accuracy(int(bestrk_uid), self.acc_type, self.args.valid)
        except:
            rk_maxacc = self.searchspace.get_final_accuracy(bestrk_uid, self.acc_type, self.args.valid)
        
        ind_rk = [totrk[uid] for uid in indices]

        return rk_ni[-1], rk_naswot[-1], rk_logsynflow[-1], bestrk_uid, maxacc, rk_maxacc, ind_rk
