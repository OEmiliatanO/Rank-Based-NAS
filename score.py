import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import importlib
import os
import sys
from tqdm import trange, tqdm
import time
from utils import add_dropout, remap_dataset_names, parameter_count
from score import *
from scipy.stats import kendalltau
from Parser import parser
from encoder import encoder

args = parser.score_argsparser()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("score networks")
print(f"Use GPU {args.GPU}...")
device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
print(f"The current device used is {device}.")

print(f"Initialize the train loader...")
print(f"dataset = {args.dataset}, data location = {args.data_loc}, validation = {args.valid}")
train_loader = datasets.get_data(args.dataset, args.data_loc, args.valid, args.batch_size, args.augtype, args.repeat, args)

print(f"Initialize the nas bench api ({args.nasspace}) ...")
if args.nasspace == "nasbench201":
    args.dataset, acc_type = remap_dataset_names(args.dataset, args.valid, args.test, args.train)
else:
    acc_type = None
print(f"dataset = {args.dataset}, validation = {args.valid}")
searchspace = nasspace.get_search_space(args)

print(f"Making sure {args.save_loc} exist.\n")
os.makedirs(args.save_loc, exist_ok=True)

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

enroll_sc_fn_names = ["ni", "naswot", "logsynflow", "snip", "grasp", "parameter"]
enroll_sc_fns = {"ni": ni_score, "naswot": naswot_score, "logsynflow": logsynflow_score, "synflow": synflow_score, "ntk": ntk_score, "grasp": grasp_score, "snip": snip_score, "parameter": parameter_count}

print(f"Use score functions: {enroll_sc_fn_names} + acc")

Encoder = encoder.get_encoder(args.nasspace)
cnt = 0

def score_nets(indices, searchspace, enroll_sc_fn_names, enroll_sc_fns, W = None, args = None):
    accs = []
    scores = dict((fn_names, []) for fn_names in enroll_sc_fn_names)
    new_scores = []

    ss_n = tqdm(total = args.n_samples)
    for uid in indices:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        net = searchspace.get_network(uid, args)
        tmp = []
        for fn_name in enroll_sc_fn_names:
            sc = float(enroll_sc_fns[fn_name](net, train_loader, device, args))
            if not np.isfinite(sc): break
            tmp.append(sc)

        if len(tmp) == len(enroll_sc_fn_names):
            if W is not None:
                new_scores.append(float(torch.dot(torch.tensor(tmp, device=device), torch.tensor(W, device=device)).cpu()))
            accs.append(searchspace.get_final_accuracy(uid, acc_type, args.valid))
            for i, fn_name in enumerate(enroll_sc_fn_names):
                scores[fn_name].append(tmp[i])
        del net
        ss_n.update(1)

    if W is not None:
        return accs, scores, new_scores
    return accs, scores

codebase = Encoder.get_nrand_code(args.n_samples)
if args.nasspace == "nasbench101":
    indices = [searchspace.query_index_by_arch(searchspace.get_spec_by_arch(*Encoder.parse_code(c[0], c[1], c[2]))) for c in codebase]
elif args.nasspace == "nasbench201" or args.nasspace == "natsbenchtss":
    indices = [searchspace.get_index_by_code(Encoder.parse_code(c)) for c in codebase]
elif args.nasspace == "natsbenchsss":
    indices = [Encoder.parse_code(c) for c in codebase]

accs, scores = score_nets(indices, searchspace, enroll_sc_fn_names, enroll_sc_fns, None, args)

def standardize(L):
    std = np.std(L)
    avg = np.mean(L)
    return list(map(lambda x: float((x-avg)/std), L))

def normalize(L, a=0, b=1):
    minL = min(L)
    maxL = max(L)
    return list(map(lambda x: b*(x-minL)/(maxL-minL)+a, L))

for fn_name in enroll_sc_fn_names:
    scores[fn_name] = standardize(scores[fn_name])
    #scores[fn_name] = normalize(scores[fn_name], 0, 1)

std_accs = standardize(accs)
#std_accs = normalize(accs, 0, 1)

M = len(enroll_sc_fn_names)
corr = [[0 for j in range(M+2)] for i in range(M+2)]

# sc-sc
for i in range(M):
    for j in range(M):
        Vi = torch.tensor(scores[enroll_sc_fn_names[i]], device=device)
        Vj = torch.tensor(scores[enroll_sc_fn_names[j]], device=device)
        corr[i][j] = float(torch.sqrt(torch.dot(Vi-Vj, Vi-Vj)).cpu())

# sc-acc
W = [0 for i in range(M)]
for i in range(M):
    Vi = torch.tensor(scores[enroll_sc_fn_names[i]], device=device)
    Vj = torch.tensor(std_accs, device=device)
    corr[M][i] = corr[i][M] = float(torch.sqrt(torch.dot(Vi-Vj, Vi-Vj)).cpu())
    W[i] = float(corr[M][i] ** -1)

new_scores = []
for i in range(len(scores[enroll_sc_fn_names[0]])):
    new_score = 0
    for j in range(M):
        new_score += scores[enroll_sc_fn_names[j]][i] * W[j]
    new_scores.append(new_score)

std_new_scores = standardize(new_scores)

# new_sc-sc
for i in range(M):
    Vi = torch.tensor(scores[enroll_sc_fn_names[i]], device=device)
    Vj = torch.tensor(std_new_scores, device=device)
    corr[M+1][i] = corr[i][M+1] = float(torch.sqrt(torch.dot(Vi-Vj, Vi-Vj)).cpu())
# new_sc-acc
Vi = torch.tensor(std_accs, device=device)
corr[M][M+1] = corr[M+1][M] = float(torch.sqrt(torch.dot(Vi-Vj, Vi-Vj)).cpu())

print("\n====corr====")
for i in range(M+2):
    for j in range(M+2):
        print(f"{corr[i][j]:.3f} ", end='')
    print("\n")
print("====corr====")
print(f"weights = {[(k,v) for k,v in zip(enroll_sc_fn_names, W)]}")

for sc_fn_name in enroll_sc_fn_names:
    print(f"arch acc by {sc_fn_name}: {accs[np.argmax(scores[sc_fn_name])]}")
print(f"arch acc by new_scores: {accs[np.argmax(std_new_scores)]}")
print(f"max arch acc: {max(accs)}")

# ================test================== #

random.seed(args.seed+1)
np.random.seed(args.seed+1)
torch.manual_seed(args.seed+1)

print("================test=================")
ss_n = tqdm(total = args.n_samples)
codebase = Encoder.get_nrand_code(args.n_samples)
if args.nasspace == "nasbench101":
    indices = [searchspace.query_index_by_arch(searchspace.get_spec_by_arch(*Encoder.parse_code(c[0], c[1], c[2]))) for c in codebase]
elif args.nasspace == "nasbench201" or args.nasspace == "natsbenchtss":
    indices = [searchspace.get_index_by_code(Encoder.parse_code(c)) for c in codebase]
elif args.nasspace == "natsbenchsss":
    indices = [Encoder.parse_code(c) for c in codebase]

accs2, scores2, new_scores2 = score_nets(indices, searchspace, enroll_sc_fn_names, enroll_sc_fns, W, args)

for fn_name in enroll_sc_fn_names:
    scores2[fn_name] = standardize(scores2[fn_name])
std_accs2 = standardize(accs2)
std_new_scores2 = standardize(new_scores2)

corr2 = [[0 for j in range(M+2)] for i in range(M+2)]

# sc-sc
for i in range(M):
    for j in range(M):
        Vi = torch.tensor(scores2[enroll_sc_fn_names[i]], device=device)
        Vj = torch.tensor(scores2[enroll_sc_fn_names[j]], device=device)
        corr2[i][j] = float(torch.sqrt(torch.dot(Vi-Vj, Vi-Vj)).cpu())

# sc-accs
for i in range(M):
    Vi = torch.tensor(scores2[enroll_sc_fn_names[i]], device=device)
    Vj = torch.tensor(std_accs2, device=device)
    corr2[M][i] = corr2[i][M] = float(torch.sqrt(torch.dot(Vi-Vj, Vi-Vj)).cpu())

# new_sc-sc
for i in range(M):
    Vi = torch.tensor(scores2[enroll_sc_fn_names[i]], device=device)
    Vj = torch.tensor(std_new_scores2, device=device)
    corr2[M+1][i] = corr2[i][M+1] = float(torch.sqrt(torch.dot(Vi-Vj, Vi-Vj)).cpu())

# new_sc-acc
Vi = torch.tensor(std_accs2, device=device)
corr2[M][M+1] = corr2[M+1][M] = float(torch.sqrt(torch.dot(Vi-Vj, Vi-Vj)).cpu())

print("\n====corr2====")
for i in range(M+2):
    for j in range(M+2):
        print(f"{corr2[i][j]:.3f} ", end='')
    print("\n")
print("====corr2====")

for sc_fn_name in enroll_sc_fn_names:
    print(f"arch acc by {sc_fn_name}: {accs2[np.argmax(scores2[sc_fn_name])]}")
print(f"arch acc by new_scores: {accs2[np.argmax(new_scores2)]}")
print(f"max arch acc: {max(accs2)}")

state = {'ni': scores["ni"],
         'ni2': scores2["ni"],
         'naswot': scores["naswot"],
         'naswot2': scores2["naswot"],
         'logsynflow': scores["logsynflow"],
         'logsynflow2': scores2["logsynflow"],
         "snip": scores["snip"], 
         "snip2": scores2["snip"], 
         "grasp": scores["grasp"], 
         "grasp2": scores2["grasp"], 
         "parameter": scores["parameter"], 
         "parameter2": scores2["parameter"], 
         "new_scores": new_scores, 
         "new_scores2": new_scores2, 
         "std_new_scores": std_new_scores, 
         "std_new_scores2": std_new_scores2, 
         "W": W, 
         "accs": accs, 
         "accs2": accs2, 
         "std_accs": std_accs, 
         "std_accs2": std_accs2, 
         'corr': corr, 
         'corr2': corr2
         }

fname = f"{args.save_loc}/{args.save_string}_{args.nasspace}_{args.dataset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.valid}_{args.test}_{args.seed}.t7"
torch.save(state, fname)
