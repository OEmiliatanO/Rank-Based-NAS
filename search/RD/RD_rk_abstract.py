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
        self.enrolled_fn_names = []
        self.enrolled_fns = {}

    def enroll(self, fn_name, fn):
        self.enrolled_fn_names.append(fn_name)
        self.enrolled_fns[fn_name] = fn

    def ranking2(self, indices, rk_fn_names):
        for fn_name in rk_fn_names:
            if fn_name not in self.enrolled_fn_names:
                assert False, f"{fn_name} is not in enrolled score functions. Aborting."

        scores = {}
        times = {}
        taus = {}
        best_uids = {}
        for fn_name in self.enrolled_fn_names:
            scores[fn_name] = []
            times[fn_name] = 0
            taus[fn_name] = np.nan
            best_uids[fn_name] = -1
        scores["rank"] = [0] * len(indices)
        times["rank"] = 0
        taus["rank"] = np.nan
        best_uids["rank"] = -1

        for uid in indices:
            if not isinstance(uid, str):
                uid = int(uid)
            network = self.searchspace.get_network(uid, self.args)
            network = network.to(self.device)

            for fn_name in self.enrolled_fn_names:
                t0 = time.time()
                scores[fn_name].append(self.enrolled_fns[fn_name](network, self.train_loader, self.device, self.args))
                times[fn_name] += time.time() - t0
                if fn_name in rk_fn_names:
                    times["rank"] += times[fn_name]

            del network
        
        t0 = time.time()
        
        if not isinstance(indices[0], str):
            accs = [self.searchspace.get_final_accuracy(int(uid), self.acc_type, self.args.valid) for uid in indices]
        else:
            accs = [self.searchspace.get_final_accuracy(uid, self.acc_type, self.args.valid) for uid in indices]
        maxacc = max(accs)

        for fn_name in self.enrolled_fn_names:
            m = np.argsort([*enumerate(scores[fn_name])], axis=0)
            best_uids[fn_name] = indices[m[-1][1]]
            valid_m = np.isfinite(scores[fn_name])
            taus[fn_name], p = kendalltau(np.array(scores[fn_name])[valid_m], np.array(accs)[valid_m])
            if fn_name in rk_fn_names:
                for ind_rk in m:
                    ind = ind_rk[1]
                    rk = ind_rk[0]
                    scores["rank"][ind] += len(indices) - rk

        times["rank"] += time.time() - t0
        best_uids["rank"] = indices[np.argmin(scores["rank"])]
        rk_maxacc = accs[np.argmin(scores["rank"])]
        taus["rank"], p = kendalltau(np.array(scores["rank"]), np.array(accs))

        return best_uids, taus, maxacc, rk_maxacc, times

    def ranking(self, indices, weight, cnt):
        rk_st = time.time()
        scores = {"ni": [], "naswot": [], "logsynflow": [], "synflow": [], "ntk": []}
        
        ni_time = 0
        naswot_time = 0
        logsyn_time = 0
        syn_time = 0
        ntk_time = 0
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
            
            syn_st = time.time()
            scores["synflow"].append(synflow_score(network, self.train_loader, self.device))
            syn_time += time.time() - syn_st + overhead
            
            ntk_st = time.time()
            scores["ntk"].append(ntk_score(network, self.train_loader, self.device))
            ntk_time += time.time() - ntk_st + overhead

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
        
        m_syn = np.argsort(scores["synflow"])
        rk_syn = indices[m_syn]
        syn_tau, p = kendalltau(np.array(scores["synflow"])[np.isfinite(scores["synflow"])], np.array(accs)[np.isfinite(scores["synflow"])])
        
        m_ntk = np.argsort(scores["ntk"])
        rk_ntk = indices[m_ntk]
        ntk_tau, p = kendalltau(np.array(scores["ntk"])[np.isfinite(scores["ntk"])], np.array(accs)[np.isfinite(scores["ntk"])])

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

