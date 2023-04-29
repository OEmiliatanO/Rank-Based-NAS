import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
import time
import sys
from tqdm import tqdm, trange
from torch import nn
from scipy import stats
from encoder import encoder
from pycls.models.nas.nas import Cell
from utils import add_dropout, init_network
from statistics import mean
from scipy.stats import kendalltau
from score import *

args = parser.score_argsparser()

print(f"Use GPU {args.GPU}")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Current device used is {device}")

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
    assert False, "Unknown dataset: {args.dataset}"

print(f"Initialize the train loader...")
train_loader = datasets.get_data(args.dataset, args.data_loc, args.valid, args.batch_size, args.augtype, args.repeat, args)

print(f"Initialize the nasspace...")
args.dataset, acc_type = remap_dataset_names(args.dataset, args.valid, args.test, args.train)
print(f"dataset = {args.dataset}, valid = {args.valid}, test = {args.test}")
searchspace = nasspace.get_search_space(args)

os.makedirs(args.save_loc, exist_ok=True)

filename_ninaswot = f'{args.save_loc}/ninaswot_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}'
filename_ni       = f'{args.save_loc}/ni_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}'
filename_ntk      = f'{args.save_loc}/ntk_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}'
filename_logsynflow = f'{args.save_loc}/logsynflow_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}'
filename_acc = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.valid}'

filenames = {"ninaswot": filename_ninaswot, "ni": filename_ni, "ntk": filename_ntk, "logsynflow": filename_logsynflow, "acc": filename_acc}
print(f"Files to save: {filenames}")

scores = dict(zip(["ninaswot", "ni", "ntk", "synflow", "logsynflow"], [np.full(len(searchspace), np.nan) for i in range(5)]))
accs = np.full(len(searchspace), np.nan)

#print(f"Start calculating means and stds in {args.n_samples} samples...")
#means, stds = get_mean_std(searchspace, args.n_samples, train_loader, device, args)
#means["ninaswot"] = 0
#stds["ninaswot"]  = np.sqrt(5)
#print(f"Done")
#print(f"means = {means}")
#print(f"stds  = {stds}")

#print(f"Start scoring the whole arches...")

if args.nasspace == 'nasbench201':
    Encoder = encoder.nasbench201_encoder()
elif args.nasspace == 'natsbenchSSS':
    Encoder = encoder.natsbenchSSS_encoder()
else:
    assert False, f"no such searchspace:{args.nasspace}"

"""
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
broken_code = np.array([2,1,4,1,4,2,2,2,4,2])
uid = searchspace.get_index_by_code(Encoder.parse_code(broken_code))
network = searchspace.get_network(uid)
#print(network)
print("uid = ", uid)
print("code = ", broken_code)
print("after parse code = ", Encoder.parse_code(broken_code))
print("score = ", ni_score(network, train_loader, device, args, uid, broken_code))
sys.exit(0)
"""

codebase = Encoder.get_nrand_code(args.n_samples)
arches = np.array([searchspace.get_index_by_code(Encoder.parse_code(c)) for c in codebase])

#nruns = tqdm(total = len(searchspace))
nruns = tqdm(total = len(arches))
times = []
#for i, (uid, network) in enumerate(searchspace):
for i, uid in enumerate(arches):
    st = time.time()
    try:
        network = searchspace.get_network(uid)
        network = network.to(device)
        #standardize = lambda x, m, s: (x-m)/s
        
        # ninaswot
        #scores['ninaswot'][uid] = standardize(ninaswot_score(network, train_loader, device, stds, means, args), means["ninaswot"], stds["ninaswot"])

        # ntk
        #scores['ntk'][uid] = -standardize(ntk_score(network, train_loader, device), means["ntk"], stds["ntk"])
        
        # ni
        #scores['ni'][uid] = ni_score(network, train_loader, device, args)
        scores['ni'][i] = ni_score(network, train_loader, device, args, uid, codebase[i])

        # entropy
        #network = init_net_gaussian(network, device)
        #scores['entropy'][uid] = standardize(entropy_score(network, train_loader, device, args), means["entropy"], stds["entropy"])

        # gradsign
        #scores['gradsign'][uid] = standardize(gradsign_score(network, train_loader, device), means["gradsign"], stds["gradsign"])

        # synflow
        #scores['synflow'][uid] = standardize(synflow_score(network, train_loader, device), means["synflow"], stds["synflow"])
        
        # logsynflow
        #scores['logsynflow'][uid] = standardize(logsynflow_score(network, train_loader, device), means["logsynflow"], stds["logsynflow"])
        
        #accs[uid] = searchspace.get_final_accuracy(uid, acc_type, args.valid)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.valid)
        if i % 1000 == 0:
            pass
            #np.save(filenames['ninaswot'], scores['ninaswot'])
            np.save(filenames['ni'], scores['ni'])
            #np.save(filenames['ntk'], scores['ntk'])
            #np.save(filenames['logsynflow'], scores['logsynflow'])
            #np.save(filenames['acc'], accs)
    except Exception as e:
        print(e)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.valid)
        break
    times.append(time.time()-st)

    #maxacc_ninaswot    = accs[np.argmax(scores["ninaswot"][:uid+1])]
    #maxacc_ntk         = accs[np.argmax(scores["ntk"][:uid+1])]
    #maxacc_logsynflow  = accs[np.argmax(scores["logsynflow"][:uid+1])]
    #maxacc_ni           = accs[np.argmax(scores["ni"][:uid+1])]
    #tau, p = kendalltau(scores["ni"][:uid+1], accs[:uid+1])
    mask = np.ma.masked_invalid(scores["ni"][:i+1]).mask.astype(bool)
    maxacc_ni           = accs[np.argmax(scores["ni"][:i+1])]
    tau, p = kendalltau(scores["ni"][:i+1][~mask], accs[:i+1][~mask])
    
    #nruns.set_description(f"maxacc(ninaswot)={maxacc_ninaswot:.3f}, maxacc(ntk)={maxacc_ntk:.3f}, maxacc(logsynflow)={maxacc_logsynflow:.3f}")
    nruns.set_description(f"maxacc(ni)={maxacc_ni:.3f}, last score={scores['ni'][i]}, tau={tau}")
    nruns.update(1)

#np.save(filenames['ninaswot'], scores['ninaswot'])
np.save(filenames['ni'], scores['ni'][args.n_samples])
#np.save(filenames['logsynflow'], scores['logsynflow'])
#np.save(filenames['acc'], accs)
