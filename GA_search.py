import argparse
import nasspace
import datasets
import random
import importlib
import numpy as np
import torch
import os
import sys
from tqdm import trange
from numpy import std, mean
import time
from utils import add_dropout, remap_dataset_names
from Parser import parser

args = parser.GA_search_argsparser()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("GA search algo")
print(f"Use GPU {args.GPU}...")
device = torch.device("cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
print(f"The current device used is {device}")

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

print(f"Making sure {args.save_loc} exist.")
os.makedirs(args.save_loc, exist_ok=True)

algo_module = importlib.import_module(f"search.GA.{args.nasspace}_GA_rk")
GA = getattr(algo_module, "GA")
GA_kwargs = {"searchspace": searchspace, "train_loader": train_loader, "device": device, "args": args}
if args.nasspace == "nasbench201":
    GA_kwargs["acc_type"] = acc_type

times     = []
chosen    = []
acc       = []
topscores = []
naswot_acc = []
ni_acc = []
logsynflow_acc = []

print("\n======================================")
print(f"parameter:\nnumber of population={args.maxn_pop}\nnumber of iteration={args.maxn_iter}\nprobability of mutation={args.prob_mut}\nprobability of crossover={args.prob_cr}")
print("======================================\n\n")

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

cnt = 0
runs = trange(args.n_runs, desc='Unavailable')
for N in runs:
    start = time.time()

    sol = GA(**GA_kwargs)
    if args.verbose:
        score, acc_, uid, rk, naswotacc, niacc, logsynflowacc = sol.search()
        naswot_acc.append(naswotacc)
        ni_acc.append(niacc)
        logsynflow_acc.append(logsynflowacc)
        ninaswot_acc.append(ninaswotacc)
    else:
        score, acc_, uid, rk = sol.search()
    chosen.append(uid)
    topscores.append(score)
    acc.append(acc_)

    times.append(time.time()-start)
    if cnt == 1 or cnt % 10 == 0:
        print("")
    cnt += 1
    if args.verbose:
        runs.set_description(f"rk-acc: {mean(acc):.3f}%({std(acc):.3f}), naswot-acc: {mean(naswot_acc):.3f}%({std(naswot_acc):.3f}), ni-acc: {mean(ni_acc):.3f}%({std(ni_acc):.3f}), syn-acc: {mean(logsynflow_acc):.3f}%({std(logsynflow_acc):.3f})")
    else:
        runs.set_description(f"rk-acc: {mean(acc):.3f}%({std(acc):.3f}) time:{mean(times):.2f}")

print(f"Final mean accuracy: {np.mean(acc)}")

state = {'accs': acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

fname = f"{args.save_loc}/{args.save_string}_{args.nasspace}_{args.dataset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}_{args.valid}_{args.test}_{args.maxn_pop}_{args.maxn_iter}_{args.prob_cr}_{args.prob_mut}.t7"
torch.save(state, fname)
