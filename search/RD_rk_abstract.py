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
        scores = {"ni": [], "naswot": [], "logsynflow": [], "synflow": []}
        
        ni_time = 0
        naswot_time = 0
        #logsynflow_time = 0
        synflow_time = 0
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
            
            #logsynflow_st = time.time()
            #scores["logsynflow"].append(logsynflow_score(network, self.train_loader, self.device))
            #logsynflow_time += time.time() - logsynflow_st + overhead

            synflow_st = time.time()
            scores["synflow"].append(synflow_score(network, self.train_loader, self.device))
            synflow_time += time.time() - synflow_st + overhead

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

        m_synflow = np.argsort(scores["synflow"])
        rk_synflow = indices[m_synflow]
        bestrk = np.inf
        for rk, id in enumerate(rk_synflow):
            totrk[id] += (self.args.n_samples - rk)
            if bestrk > totrk[id]:
                bestrk = totrk[id]
                bestrk_uid = id

        #m_logsyn = np.argsort(scores["logsynflow"])
        #rk_logsynflow = indices[m_logsyn]
        #bestrk = np.inf
        #for rk, id in enumerate(rk_logsynflow):
        #    totrk[id] += (self.args.n_samples - rk)
        #    if bestrk > totrk[id]:
        #        bestrk = totrk[id]
        #        bestrk_uid = id
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
        #ind_rk = np.argsort(ind_rk)
        rk_tau, p = kendalltau(ind_rk, accs)
        ni_tau, p = kendalltau(np.array(scores["ni"])[np.isfinite(scores["ni"])], np.array(accs)[np.isfinite(scores["ni"])])
        naswot_tau, p = kendalltau(np.array(scores["naswot"])[np.isfinite(scores["naswot"])], np.array(accs)[np.isfinite(scores["naswot"])])
        #logsyn_tau, p = kendalltau(scores["logsynflow"], accs)
        synflow_tau, p = kendalltau(np.array(scores["synflow"])[np.isfinite(scores["synflow"])], np.array(accs)[np.isfinite(scores["synflow"])])
        rk_time = time.time() - rk_st
        ####
        #print(f"ni tau = {ni_tau}, naswot tau = {naswot_tau}, logsynflow tau = {logsyn_tau}, synflow tau = {synflow_tau}, rk tau = {rk_tau}")
        #np.savez("./correlation_info.npy", ind_rk = ind_rk, score_ni = scores["ni"], score_naswot = scores["naswot"], score_synflow = scores["synflow"], score_logsyn = scores["logsynflow"], acc = accs, rk_tau = np.array([rk_tau]), ni_tau = np.array([ni_tau]), naswot_tau = np.array([naswot_tau]), synflow_tau = np.array([synflow_tau]), logsyn_tau = np.array([logsyn_tau]))
        print(f"ni tau = {ni_tau}, naswot tau = {naswot_tau}, synflow tau = {synflow_tau}, rk tau = {rk_tau}")
        np.savez("./correlation_info.npy", ind_rk = ind_rk, score_ni = scores["ni"], score_naswot = scores["naswot"], score_synflow = scores["synflow"], acc = accs, rk_tau = np.array([rk_tau]), ni_tau = np.array([ni_tau]), naswot_tau = np.array([naswot_tau]), synflow_tau = np.array([synflow_tau]))
        ####
        
        bestuid = (rk_ni[-1], 
                   rk_naswot[-1], 
                   rk_synflow[-1], 
                   bestrk_uid)

        taus = (rk_tau, 
                ni_tau, 
                naswot_tau, 
                synflow_tau)

        times = (ni_time, 
                 naswot_time, 
                 synflow_time, 
                 rk_time)

        return bestuid, taus, maxacc, rk_maxacc, times

    def search(self):
        #TODO
        pass

