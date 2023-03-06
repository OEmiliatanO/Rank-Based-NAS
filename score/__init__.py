from utils import add_dropout, init_network
from .ntk_score import ntk_score
from .ninaswot_score import ninaswot_score, ni_score, naswot_score
from .entropy_score import entropy_score, init_net_gaussian
import numpy as np
from tqdm import tqdm
import time
from statistics import mean

def standardize(x, m, s):
    return (x-m)/s

def get_mean_std(searchspace, sample_n, train_loader, device, args):
    scores_naswot  = []
    scores_ni      = []
    scores_entropy = []
    scores_ntk     = []
    arches = np.random.randint(0, len(searchspace), sample_n)
    nruns = tqdm(total = sample_n)
    times = []
    for arch in arches:
        st = time.time()

        uid = searchspace[arch]
        network = searchspace.get_network(uid)

        scores_naswot.append(naswot_score(network, train_loader, device, args))
        scores_ni.append(ni_score(network, train_loader, device, args))
        scores_ntk.append(ntk_score(network, train_loader, device))
        network = init_net_gaussian(network, device)
        scores_entropy.append(entropy_score(network, train_loader, device, args))
        times.append(time.time() - st)
        nruns.set_description(f"average elapse={mean(times):.2f}")
        nruns.update(1)
        del network

    scores_naswot  = np.array(scores_naswot)
    scores_ni      = np.array(scores_ni)
    scores_ntk     = np.array(scores_ntk)
    scores_entropy = np.array(scores_entropy)
    
    calstd  = lambda x: np.ma.masked_invalid(x).std()
    calmean = lambda x: np.ma.masked_invalid(x).mean()
    stds  = {"naswot": calstd(scores_naswot), "ni": calstd(scores_ni), "entropy": calstd(scores_entropy), "ntk": calstd(scores_ntk)}
    means = {"naswot": calmean(scores_naswot), "ni": calmean(scores_ni), "entropy": calmean(scores_entropy), "ntk": calmean(scores_ntk)}
    
    del scores_naswot, scores_ni, scores_ntk, scores_entropy, times

    return means, stds

def score_tot(network, train_loader, stds, means, device, args):
    ninaswot = standardize(ninaswot_score(network, train_loader, device, stds, means, args), means["ninaswot"], stds["ninaswot"])
    ntk = -standardize(ntk_score(network, train_loader, device), means["ntk"], stds["ntk"])
    network = init_net_gaussian(network, device)
    entropy = standardize(entropy_score(network, train_loader, device, args), means["entropy"], stds["entropy"])
    tot = ninaswot + ntk + entropy
    del network
    if not np.isfinite(tot):
        return -1000000
    else:
        return tot
