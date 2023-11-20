import random
import numpy as np
import torch
import sys
from tqdm import trange
import time
from score import *
import math

class abstract_SA():
    def __init__(self, searchspace, train_loader, acc_type, device, args):
        self.end_T = args.end_T
        self.maxn_iter = args.maxn_iter
        self.Rt = args.Rt
        self.init_T = args.init_T
        self.maxN = args.maxN
        self.alpha = args.alpha
        self.beta = args.beta
        self.searchspace = searchspace
        self.train_loader = train_loader
        self.device = device
        self.acc_type = acc_type
        self.args = args
        self.enrolled_fn_names = []
        self.enrolled_fns = {}
        self.DICT = {}

    def enroll(self, fn_name, fn):
        self.enrolled_fn_names.append(fn_name)
        self.enrolled_fns[fn_name] = fn

    def neighbor(self, arch):
        #TODO
        assert False, f"Implemenation isn't completed."
        pass
    
    def rand_arch_generate(self):
        #TODO
        assert False, f"Implemenation isn't completed."
        pass

    def search(self):
        best_arch = now_arch = self.rand_arch_generate()
        now_sol_uid = best_sol_uid = self.searchspace.query_index_by_arch(self.list2arch(now_arch))
        history_best_arch = set()
        T = self.init_T
        while T > self.end_T:
            for i in range(self.maxn_iter):
                neighbors = set()
                while len(neighbors) < self.maxN:
                    neighbors.add(tuple(self.neighbor(list(now_arch))))
                neighbors.add(tuple(now_arch))

                for arch in history_best_arch:
                    neighbors.add(tuple(arch))
                
                neighbors = list(neighbors)
                now_uid = self.searchspace.query_index_by_arch(self.list2arch(now_arch))
                indices = [self.searchspace.query_index_by_arch(self.list2arch(arch)) for arch in neighbors]
                
                # E
                best_uids, maxacc, rk_maxacc, scores = self.ranking(np.array(indices))
                bestrk_uid = best_uids["rank"]
                ind_rk = scores["rank"]

                best_sol_uid = bestrk_uid
                best_arch = neighbors[indices.index(bestrk_uid)]
                history_best_arch.add(best_arch)
                # maintain the size of set H
                while len(history_best_arch) > self.beta * (self.maxn_iter * self.maxN * math.log(self.end_T / self.init_T) / math.log(self.Rt)):
                    history_best_arch = list(history_best_arch)
                    indices = [self.searchspace.query_index_by_arch(self.list2arch(arch)) for arch in history_best_arch]
                    best_uids, maxacc, rk_maxacc, scores = self.ranking(np.array(indices))
                    worst_arch_ind = np.argmax(scores["rank"])
                    del history_best_arch[worst_arch_ind]
                    history_best_arch = set(history_best_arch)

                bestrk = np.inf
                if bestrk_uid == now_uid:
                    now_ind = indices.index(now_uid)
                    for j in range(len(ind_rk)):
                        if bestrk > ind_rk[j] and j != now_ind:
                            bestrk = ind_rk[j]
                            bestrk_uid = indices[j]
                
                dE = ind_rk[indices.index(bestrk_uid)] - ind_rk[indices.index(now_uid)]
                if dE < 0 or np.log(random.uniform(0,1)) <= -self.alpha * dE / T:
                    now_uid = bestrk_uid
                    now_arch = neighbors[indices.index(bestrk_uid)]

            T *= self.Rt
            if not isinstance(best_sol_uid, str) and not isinstance(now_uid, str):
                best_sol_uid = int(best_sol_uid)
                now_uid = int(now_uid)
        return best_sol_uid, self.searchspace.get_final_accuracy(best_sol_uid, self.acc_type, self.args.valid), now_uid, self.searchspace.get_final_accuracy(now_uid, self.acc_type, self.args.valid)

    def ranking(self, indices):
        scores = {}
        best_uids = {}
        for fn_name in self.enrolled_fn_names:
            scores[fn_name] = []
            best_uids[fn_name] = -1
        scores["rank"] = [0] * len(indices)
        best_uids["rank"] = -1

        for uid in indices:
            if not isinstance(uid, str):
                uid = int(uid)

            if uid not in self.DICT:
                network = self.searchspace.get_network(uid, self.args)
                network = network.to(self.device)
                scs = []
                for fn_name in self.enrolled_fn_names:
                    sc = self.enrolled_fns[fn_name](network, self.train_loader, self.device, self.args)
                    scores[fn_name].append(sc)
                    scs.append(sc)
                self.DICT[uid] = tuple(scs)
                del network
            else:
                for i, fn_name in enumerate(self.enrolled_fn_names):
                    scores[fn_name].append(self.DICT[uid][i])

        
        if not isinstance(indices[0], str):
            accs = [self.searchspace.get_final_accuracy(int(uid), self.acc_type, self.args.valid) for uid in indices]
        else:
            accs = [self.searchspace.get_final_accuracy(uid, self.acc_type, self.args.valid) for uid in indices]
        maxacc = max(accs)

        for fn_name in self.enrolled_fn_names:
            m = np.argsort([*enumerate(scores[fn_name])], axis=0)
            best_uids[fn_name] = indices[m[-1][1]]

            for ind_rk in m:
                ind = ind_rk[1]
                rk = ind_rk[0]
                scores["rank"][ind] += len(indices) - rk

        best_uids["rank"] = indices[np.argmin(scores["rank"])]
        rk_maxacc = accs[np.argmin(scores["rank"])]

        return best_uids, maxacc, rk_maxacc, scores

