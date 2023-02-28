from utils import add_dropout, init_network
from .ntk_score import ntk_score
from .ninaswot_score import ninaswot_score, ni_score, naswot_score, get_batch_jacobian
from .entropy_score import entropy_score, init_net_gaussian
import numpy as np

def get_mean_std(searchspace, sample_n, train_loader, device, args):
    scores_naswot  = []
    scores_ni      = []
    scores_entropy = []
    scores_ntk     = []
    arches = np.random.randint(0, len(searchspace), sample_n)
    for arch in arches:
        uid = searchspace[arch]
        network = searchspace.get_network(uid)

        scores_naswot.append(naswot_score(network, train_loader, device, args))
        scores_ni.append(ni_score(network, train_loader, device, args))
        scores_ntk.append(ntk_score(network, train_loader, device))
        network = init_net_gaussian(network, device)
        scores_entropy.append(entropy_score(network, train_loader, device, args))

    scores_naswot  = np.array(scores_naswot)
    scores_ni      = np.array(scores_ni)
    scores_ntk     = np.array(scores_ntk)
    scores_entropy = np.array(scores_entropy)
    
    calstd  = lambda x: np.ma.masked_invalid(x).std()
    calmean = lambda x: np.ma.masked_invalid(x).mean()
    stds  = {"naswot": calstd(scores_naswot), "ni": calstd(scores_ni), "entropy": calstd(scores_entropy), "ntk": calstd(scores_ntk)}
    means = {"naswot": calmean(scores_naswot), "ni": calmean(scores_ni), "entropy": calmean(scores_entropy), "ntk": calmean(scores_ntk)}
    
    del scores_naswot, scores_ni, scores_ntk, scores_entropy

    return stds, means
