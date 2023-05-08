import random
import numpy as np
import torch
import os
import sys
from score import *
from scipy.stats import kendalltau
from weight_giver import Weight_giver

class abstract_RD():
    def __init__(self, searchspace, train_loader, acc_type, Encoder, device, args):
        self.searchspace = searchspace
        self.train_loader = train_loader
        self.device = device
        self.args = args
        self.acc_type = acc_type
        self.Encoder = Encoder

    def ranking(self, indices, weight, cnt):
        scores = {"ni": [], "naswot": [], "logsynflow": []}
        
        for uid in indices:
            uid = int(uid)
            network = self.searchspace.get_network(uid, self.args)
            network = network.to(self.device)
            scores["ni"].append(ni_score(network, self.train_loader, self.device, self.args))
            scores["naswot"].append(naswot_score(network, self.train_loader, self.device, self.args))
            scores["logsynflow"].append(logsynflow_score(network, self.train_loader, self.device))
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
        
        accs = [self.searchspace.get_final_accuracy(int(uid), self.acc_type, self.args.valid) for uid in indices]
        maxacc = np.max(accs)
        rk_maxacc = self.searchspace.get_final_accuracy(int(bestrk_uid), self.acc_type, self.args.valid)
        
        ind_rk = [totrk[uid] for uid in indices]
        rk_tau, p = kendalltau(ind_rk, accs)
        ni_tau, p = kendalltau(scores["ni"], accs)
        naswot_tau, p = kendalltau(scores["naswot"], accs)
        logsyn_tau, p = kendalltau(scores["logsynflow"], accs)

        return rk_ni[-1], rk_naswot[-1], rk_logsynflow[-1], bestrk_uid, rk_tau, ni_tau, naswot_tau, logsyn_tau, maxacc, rk_maxacc

    def search(self):
        #TODO
        pass

