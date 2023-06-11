import random
import numpy as np
import torch
import os
import sys
from score import *
from scipy.stats import kendalltau
import time

class abstract_RD():
    def __init__(self, searchspace, train_loader, acc_type, Encoder, device, args):
        self.searchspace = searchspace
        self.train_loader = train_loader
        self.device = device
        self.args = args
        self.acc_type = acc_type
        self.Encoder = Encoder

    def ranking(self, indices, weight, cnt):
        rk_st = time.time()
        scores = {"ni": [], "naswot": [], "logsynflow": []}
        
        ni_time = 0
        naswot_time = 0
        logsyn_time = 0
        for uid in indices:
            overhead_st = time.time()
            try:
                uid = int(uid)
            except:
                pass
            network = self.searchspace.get_network(uid, self.args)
            network = network.to(self.device)
            overhead = time.time() - overhead_st

            ni_st = time.time()
            scores["ni"].append(ni_score(network, self.train_loader, self.device, self.args))
            ni_time += time.time() - ni_st + overhead
            
            naswot_st = time.time()
            scores["naswot"].append(naswot_score(network, self.train_loader, self.device, self.args))
            naswot_time += time.time() - naswot_st + overhead
            
            logsyn_st = time.time()
            scores["logsynflow"].append(logsynflow_score(network, self.train_loader, self.device))
            logsyn_time += time.time() - logsyn_st + overhead

            del network

        totrk = dict(zip([uid for uid in indices], [0 for i in range(self.args.n_samples)]))
        
        m_ni = np.argsort(scores["ni"])
        rk_ni = indices[m_ni]
        for rk, id in enumerate(rk_ni):
            totrk[id] += (self.args.n_samples - rk)
        
        m_naswot = np.argsort(scores["naswot"])
        rk_naswot = indices[m_naswot]
        for rk, id in enumerate(rk_naswot):
            totrk[id] += (self.args.n_samples - rk)

        m_logsyn = np.argsort(scores["logsynflow"])
        rk_logsyn = indices[m_logsyn]
        bestrk = np.inf
        for rk, id in enumerate(rk_logsyn):
            totrk[id] += (self.args.n_samples - rk)
            if bestrk > totrk[id]:
                bestrk = totrk[id]
                bestrk_uid = id
        try:
            accs = [self.searchspace.get_final_accuracy(int(uid), self.acc_type, self.args.valid) for uid in indices]
        except:
            accs = [self.searchspace.get_final_accuracy(uid, self.acc_type, self.args.valid) for uid in indices]
        maxacc = np.max(accs)
        try:
            bestrk_uid = int(bestrk_uid)
        except:
            pass
        rk_maxacc = self.searchspace.get_final_accuracy(bestrk_uid, self.acc_type, self.args.valid)
        
        ind_rk = [totrk[uid] for uid in indices]
        rk_tau, p = kendalltau(ind_rk, accs)
        ni_tau, p = kendalltau(np.array(scores["ni"])[np.isfinite(scores["ni"])], np.array(accs)[np.isfinite(scores["ni"])])
        naswot_tau, p = kendalltau(np.array(scores["naswot"])[np.isfinite(scores["naswot"])], np.array(accs)[np.isfinite(scores["naswot"])])
        logsyn_tau, p = kendalltau(np.array(scores["logsynflow"])[np.isfinite(scores["logsynflow"])], np.array(accs)[np.isfinite(scores["logsynflow"])])
        rk_time = time.time() - rk_st
        
        bestuid = (rk_ni[-1], 
                   rk_naswot[-1], 
                   rk_logsyn[-1], 
                   bestrk_uid)

        taus = (rk_tau, 
                ni_tau, 
                naswot_tau, 
                logsyn_tau)

        times = (ni_time, 
                 naswot_time, 
                 logsyn_time, 
                 rk_time)

        return bestuid, taus, maxacc, rk_maxacc, times

    def search(self):
        #TODO
        pass

