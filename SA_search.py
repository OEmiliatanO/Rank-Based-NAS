import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
import sys
import importlib
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from numpy import mean, std
import time
from utils import add_dropout, remap_dataset_names
from Parser import parser
from score import *

args = parser.SA_search_argsparser()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("SA search algo")
print(f"Use GPU {args.GPU}...")
device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
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

algo_module = importlib.import_module(f"search.SA.{args.nasspace}_SA_rk")
SA = getattr(algo_module, "SA")
SA_kwargs = {"searchspace": searchspace, "train_loader": train_loader, "acc_type": None, "device": device, "args": args}
if args.nasspace == "nasbench201":
    SA_kwargs["acc_type"] = acc_type

times     = []
chosen    = []
acc       = []
print("\n=====================================")
print(f"parameter:\nendT = {args.end_T}\nmaxIter = {args.maxn_iter}\nRt = {args.Rt}\ninit_T = {args.init_T}\nmaxN = {args.maxN}\nalpha = {args.alpha}\nbeta = {args.beta}")
print("=====================================\n\n")

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

acc = []
enroll_sc_fn_names = ["ni", "naswot", "logsynflow"]
enroll_sc_fns = {"ni": ni_score, "naswot": naswot_score, "logsynflow": logsynflow_score}
print(f"score functions used in SA: {enroll_sc_fn_names}")

cnt = 0
runs = trange(args.n_runs, desc='acc: ')
for N in runs:
    start = time.time()

    alg = SA(**SA_kwargs)
    for fn_name in enroll_sc_fn_names:
        alg.enroll(fn_name = fn_name, fn = enroll_sc_fns[fn_name])

    best_sol_uid, rk_maxacc, current_uid, rk_current_acc = alg.search()
    
    chosen.append(current_uid)
    acc.append(rk_current_acc)

    times.append(time.time()-start)
    if cnt == 1 or cnt % 10 == 0:
        print("")
    cnt += 1
    runs.set_description(f"acc: {mean(acc):.2f}({std(acc):.2f}), time:{mean(times):.2f}")

print(f"Final mean test accuracy: {np.mean(acc)}")

state = {'accs': acc,
         'chosen': chosen,
         'times': times,
         }

fname = f"{args.save_loc}/{args.save_string}_{args.nasspace}_{args.dataset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}_{args.valid}_{args.test}_{args.end_T}_{args.maxn_iter}_{args.Rt}_{args.init_T}_{args.maxN}.t7"
torch.save(state, fname)
