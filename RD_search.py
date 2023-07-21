import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
from torch import nn
import importlib
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from statistics import mean
from numpy import std
import time
from utils import add_dropout, remap_dataset_names
from score import *
from encoder import encoder
from scipy.stats import kendalltau
from Parser import parser

args = parser.RD_search_argsparser()

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("RD search algo")
print(f"Use GPU {args.GPU}...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

print(f"Making sure {args.save_loc} exist.\n")
os.makedirs(args.save_loc, exist_ok=True)


Encoder = encoder.get_encoder(args.nasspace)

algo_module = importlib.import_module(f"search.RD.{args.nasspace}_RD_rk")
RD = getattr(algo_module, "RD")
RD_kwargs = {"searchspace": searchspace, "train_loader": train_loader, "device": device, "args": args, "acc_type": None, "Encoder": Encoder}
if args.nasspace == "nasbench201":
    RD_kwargs["acc_type"] = acc_type

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

enroll_sc_fn_names = ["ni", "naswot", "logsynflow", "synflow", "ntk"]
enroll_sc_fns = {"ni": ni_score, "naswot": naswot_score, "logsynflow": logsynflow_score, "synflow": synflow_score, "ntk": ntk_score}
accs = dict(zip(enroll_sc_fn_names, [[]] * len(enroll_sc_fn_names)))
taus = dict(zip(enroll_sc_fn_names, [[]] * len(enroll_sc_fn_names)))
times = dict(zip(enroll_sc_fn_names, [[]] * len(enroll_sc_fn_names)))
accs["rank"] = []
taus["rank"] = []
times["rank"] = []
bestacc = []

cnt = 0
runs = trange(args.n_runs, desc='RS algorithm')
for N in runs:
    start = time.time()

    alg = RD(**RD_kwargs)
    for fn_name in enroll_sc_fn_names:
        alg.enroll(fn_name, enroll_sc_fns[fn_name])

    rank_alg_fns = ["ni", "naswot", "logsynflow"]

    best_uids_, taus_, maxacc, rk_maxacc, times_ = alg.search(rank_alg_fns)

    for fn_name in enroll_sc_fn_names:
        if not isinstance(best_uids_[fn_name], str):
            best_uids_[fn_name] = int(best_uids_[fn_name])
        accs[fn_name].append(searchspace.get_final_accuracy(best_uids_[fn_name], acc_type, args.valid))
        taus[fn_name].append(taus_[fn_name])
        times[fn_name].append(times_[fn_name])

    accs["rank"].append(rk_maxacc)
    taus["rank"].append(taus_["rank"])
    times["rank"].append(times_["rank"])

    bestacc.append(maxacc)

    if cnt == 1 or cnt % 10 == 0:
        print("")
    cnt += 1

    info_ = ""
    for fn_name in enroll_sc_fn_names:
        info_ += f"{fn_name}: {mean(accs[fn_name]):.2f}({std(accs[fn_name]):.2f}), tau: {mean(taus[fn_name]):.2f}, time:{mean(times[fn_name]):.2f}\n"
    info_ += f"rank: {mean(accs['rank']):.2f}({std(accs['rank']):.2f}), tau: {mean(taus['rank']):.2f}, time:{mean(times['rank']):.2f}\n"

    runs.set_description(info_)

"""
state = {'ni-accs': accs["ni"],
         'naswot': accs["naswot"],
         'logsynflow': accs["logsynflow"],
         'rank-based': accs["rank-based"],
         'tot-times': times["tot"],
         'ni-times': times["ni"],
         'naswot-times': times["naswot"],
         'logsynflow-times': times["logsynflow"],
         'rk-times': times["rk"],
         }

fname = f"{args.save_loc}/{args.save_string}_{args.nasspace}_{args.dataset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.valid}_{args.test}_{args.seed}.t7"
torch.save(state, fname)
"""
