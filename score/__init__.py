from utils import add_dropout, init_network

from .ntk_score import ntk_score
from .ninaswot_score import ninaswot_score, ni_score, naswot_score
from .synflow_score import synflow_score
from .logsynflow_score import logsynflow_score

import numpy as np
from tqdm import tqdm
import time
from statistics import mean

def standardize(x, m, s):
    return (x-m)/s

def get_mean_std(searchspace, sample_n, train_loader, device, args):
    scores_naswot   = []
    scores_ni       = []
    scores_ntk      = []
    scores_logsynflow  = []
    arches = np.random.randint(0, len(searchspace), sample_n)
    nruns = tqdm(total = sample_n)
    times = []
    for arch in arches:
        st = time.time()

        uid = searchspace[arch]
        network = searchspace.get_network(uid)

        scores_naswot.append(naswot_score(network, train_loader, device, args))
        scores_ni.append(ni_score(network, train_loader, device, args))
        #scores_ntk.append(ntk_score(network, train_loader, device))
        scores_ntk.append(0)
        scores_logsynflow.append(logsynflow_score(network, train_loader, device))
        
        times.append(time.time() - st)
        nruns.set_description(f"average elapse={mean(times):.2f}")
        nruns.update(1)
        del network

    scores_naswot   = np.array(scores_naswot)
    scores_ni       = np.array(scores_ni)
    scores_ntk      = np.array(scores_ntk)
    scores_logsynflow  = np.array(scores_logsynflow)
    
    
    calstd  = lambda x: np.ma.masked_invalid(x).std()
    calmean = lambda x: np.ma.masked_invalid(x).mean()
    stds  = {"naswot": calstd(scores_naswot), "ni": calstd(scores_ni), "ntk": calstd(scores_ntk), "ninaswot": np.sqrt(5), "logsynflow": calstd(scores_logsynflow)}
    means = {"naswot": calmean(scores_naswot), "ni": calmean(scores_ni), "ntk": calmean(scores_ntk), "ninaswot": 0, "logsynflow": calmean(scores_logsynflow)}
    
    del scores_naswot, scores_ni, scores_ntk, scores_logsynflow, times

    return means, stds
