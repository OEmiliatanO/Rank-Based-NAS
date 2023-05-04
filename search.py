import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
from torch import nn
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from statistics import mean
from numpy import std
import time
from utils import add_dropout
from score import *
from encoder import encoder
from scipy.stats import kendalltau
from Parser import parser
from weight_giver import Weight_giver

args = parser.search_argsparser()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print(f"Use GPU {args.GPU}...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The current device used is {device}")

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def remap_dataset_names(dataset, valid, test, train):
    cifar10 = 'cifar10'
    if dataset == cifar10 and valid:
        return cifar10 + '-valid', 'x-valid'
    if dataset == cifar10 and test:
        return cifar10, 'ori-test'
    if dataset == cifar10 and train:
        return cifar10, 'train'

    assert not train, "no train label"
    cifar100 = 'cifar100'
    if dataset == cifar100 and valid:
        return cifar100, 'x-valid'
    if dataset == cifar100 and test:
        return cifar100, 'x-test'

    ImageNet16_120 = 'ImageNet16-120'
    if dataset == ImageNet16_120 and valid:
        return ImageNet16_120, 'x-valid'
    if dataset == ImageNet16_120 and test:
        return ImageNet16_120, 'x-test'
    assert False, "Unknown dataset {args.dataset}"

print(f"Initialize the train loader...")
print(f"dataset = {args.dataset}, data location = {args.data_loc}, validation = {args.valid}")
train_loader = datasets.get_data(args.dataset, args.data_loc, args.valid, args.batch_size, args.augtype, args.repeat, args)

print(f"Initialize the nas bench api {args.nasspace} ...")
if args.nasspace != "natsbenchSSS":
    args.dataset, acc_type = remap_dataset_names(args.dataset, args.valid, args.test, args.train)
else:
    acc_type = None
print(f"dataset = {args.dataset}, validation = {args.valid}")
searchspace = nasspace.get_search_space(args)

print(f"Making sure {args.save_loc} exist.")
os.makedirs(args.save_loc, exist_ok=True)

"""
print(f"Currently calculate means and standards.")
means, stds = get_mean_std(searchspace, 100, train_loader, device, args)
print(f"Calculation of means and stds is done.")
print(f"means = {means}\nstds = {stds}")
print(f"========================================")
"""

times = []
accs = {"ni": [], "naswot": [], "logsynflow": [], "rank-based": []}

Encoder = encoder.get_encoder(args.nasspace)

def ranking(indices, weight, cnt):
    scores = {"ni": [], "naswot": [], "logsynflow": []}
    
    for uid in indices:
        uid = int(uid)
        network = searchspace.get_network(uid, args)
        #print(uid)
        network = network.to(device)
        scores["ni"].append(ni_score(network, train_loader, device, args))
        scores["naswot"].append(naswot_score(network, train_loader, device, args))
        scores["logsynflow"].append(logsynflow_score(network, train_loader, device))
        del network

    totrk = dict(zip([uid for uid in indices], [0 for i in range(args.n_samples)]))
    
    m_ni = np.argsort(scores["ni"])
    rk_ni = indices[m_ni]
    for rk, id in enumerate(rk_ni):
        totrk[id] += (args.n_samples - rk) * weight[0]
    
    m_naswot = np.argsort(scores["naswot"])
    rk_naswot = indices[m_naswot]
    for rk, id in enumerate(rk_naswot):
        totrk[id] += (args.n_samples - rk) * weight[1]

    m_logsyn = np.argsort(scores["logsynflow"])
    rk_logsynflow = indices[m_logsyn]
    bestrk = np.inf
    for rk, id in enumerate(rk_logsynflow):
        totrk[id] += (args.n_samples - rk) * weight[2]

        if bestrk > totrk[id]:
            bestrk = totrk[id]
            bestrk_uid = id
    
    accs = [searchspace.get_final_accuracy(int(uid), acc_type, args.valid) for uid in indices]
    maxacc = np.max(accs)
    rk_maxacc = searchspace.get_final_accuracy(int(bestrk_uid), acc_type, args.valid)
    
    ind_rk = [totrk[uid] for uid in indices]
    rk_tau, p = kendalltau(ind_rk, accs)
    ni_tau, p = kendalltau(scores["ni"], accs)
    naswot_tau, p = kendalltau(scores["naswot"], accs)
    logsyn_tau, p = kendalltau(scores["logsynflow"], accs)
    if args.save:
        filename_rk = f'{args.save_loc}/rk_{args.nasspace}_{args.dataset}_{args.n_runs}_{args.n_samples}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.seed}_{cnt}'
        filename_acc = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.valid}_{cnt}'
        np.save(filename_rk, ind_rk)
        np.save(filename_acc, accs)
    return rk_ni[-1], rk_naswot[-1], rk_logsynflow[-1], bestrk_uid, rk_tau, ni_tau, naswot_tau, logsyn_tau, maxacc, rk_maxacc

##
"""
weight_giver = Weight_giver(args.n_samples, 6, 3, 1)
weight_giver.to(device)

for _ in range(50):
    codebase = Encoder.get_nrand_code(args.n_samples)
    indices = np.array([searchspace.get_index_by_code(Encoder.parse_code(c)) for c in codebase])
    arch_str = [Encoder.parse_code(c) for c in codebase]
    weight_giver.train(arch_str, indices, ranking, nn.CrossEntropyLoss(), weight_giver.optimizer, device)
"""
##
"""
codebase = Encoder.get_nrand_code(args.n_samples)
code = codebase[5]
indices = np.array([Encoder.parse_code(c) for c in codebase])
np.save("debug-idsss.npy", indices)
accs = [searchspace.get_final_accuracy(int(uid), acc_type, args.valid) for uid in indices]
np.save("debug-accsss.npy", accs)
ni_scores = []
nruns = tqdm(total = len(indices))
for uid in indices:
    network = searchspace.get_network(int(uid), args)
    ni_scores.append(ni_score(network, train_loader, device, args))
    nruns.update(1)

np.save("debug-nisss.npy", ni_scores)
tau, p = kendalltau(ni_scores, accs)
print(f"tau = {tau}")
"""
"""
print("code = ", code)
id = Encoder.parse_code(code)
network = searchspace.get_network(id, args)
print(f"after parse: {id}")
network = network.to(device)
print(f"score = {ni_score(network, train_loader, device, args)}")
print(f"acc = {searchspace.get_final_accuracy(id, acc_type, args.valid)}")
"""
#sys.exit(0)

taus = {"rk":[], "ni": [], "naswot": [], "logsynflow": []}
cnt = 0
runs = trange(args.n_runs, desc='acc: ')
for N in runs:
    start = time.time()
    #indices = np.random.randint(0,len(searchspace),args.n_samples)
    codebase = Encoder.get_nrand_code(args.n_samples)
    if args.nasspace == "nasbench201":
        indices = np.array([searchspace.get_index_by_code(Encoder.parse_code(c)) for c in codebase])
        archstrs = [searchspace.get_arch_str_by_code(c) for c in codebase]
    elif args.nasspace == "natsbenchSSS":
        indices = np.array([Encoder.parse_code(c) for c in codebase])
    elif args.nasspace == "nasbench101":
        pass

    #w = weight_giver(archstrs)
    niuid, naswotuid, logsynuid, bestrk_uid, rk_tau, ni_tau, naswot_tau, logsyn_tau, _, __= ranking(indices, [1,1,1], cnt)
    cnt += 1
    taus['rk'].append(rk_tau)
    taus['ni'].append(ni_tau)
    taus['naswot'].append(naswot_tau)
    taus['logsynflow'].append(logsyn_tau)

    accs["ni"].append(searchspace.get_final_accuracy(int(niuid), acc_type, args.valid))
    accs["naswot"].append(searchspace.get_final_accuracy(int(naswotuid), acc_type, args.valid))
    accs["logsynflow"].append(searchspace.get_final_accuracy(int(logsynuid), acc_type, args.valid))
    accs["rank-based"].append(searchspace.get_final_accuracy(int(bestrk_uid), acc_type, args.valid))

    times.append(time.time()-start)
    runs.set_description(f"acc_ni: {mean(accs['ni']):.2f}({std(accs['ni']):.3f}),{mean(taus['ni']):.3f}({std(taus['ni']):.3f}), acc_naswot: {mean(accs['naswot']):.2f}({std(accs['naswot']):.3f}),{mean(taus['naswot']):.3f}({std(taus['naswot']):.3f}), acc_logsyn: {mean(accs['logsynflow']):.2f}({std(accs['logsynflow']):.3f}),{mean(taus['logsynflow']):.3f}({std(taus['logsynflow']):.3f}), rk-based: {mean(accs['rank-based']):.2f}({std(accs['rank-based']):.3f}),{mean(taus['rk']):.3f}({std(taus['rk']):.3f})")

"""
state = {'accs': acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

dset = args.dataset if not (args.trainval and args.dataset == 'cifar10') else 'cifar10-valid'
fname = f"{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{dset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"
torch.save(state, fname)
"""
