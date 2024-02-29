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
accs = dict((fn_names, []) for fn_names in enroll_sc_fn_names)
taus = dict((fn_names, []) for fn_names in enroll_sc_fn_names)
times = dict((fn_names, []) for fn_names in enroll_sc_fn_names)
scores = dict((fn_names, []) for fn_names in enroll_sc_fn_names)

print(f"Use score functions: {enroll_sc_fn_names}")

Encoder = encoder.get_encoder(args.nasspace)
cnt = 0
for N in range(args.n_runs):
    #ss_n = tqdm(total = len(searchspace))
    ss_n = tqdm(total = args.n_samples)
    codebase = Encoder.get_nrand_code(args.n_samples)
    indices = [Encoder.parse_code(c) for c in codebase]
    for uid in indices:
        # Reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
        net = searchspace.get_network(uid, args)
        tmp = []
        for fn_name in enroll_sc_fn_names:
            sc = enroll_sc_fns[fn_name](net, train_loader, device, args)
            if not np.isfinite(sc): break
            tmp.append(sc)

        if len(tmp) == len(enroll_sc_fn_names):
            for i, fn_name in enumerate(enroll_sc_fn_names):
                scores[fn_name].append(tmp[i])
                accs[fn_name].append(searchspace.get_final_accuracy(uid, acc_type, args.valid))
                if not np.isfinite(scores[fn_name][-1]):
                    print(f"{fn_name}: {uid}, {scores[fn_name][-1]}")
        del net
        ss_n.update(1)

    info_ = ""
    for fn_name in enroll_sc_fn_names:
        info_ += f"{fn_name}: "
        info_ += " " * (10 - len(fn_name))
        info_ += f"{np.mean(accs[fn_name]):.2f}({np.std(accs[fn_name]):.2f})\n"

for fn_name in enroll_sc_fn_names:
    std = np.std(scores[fn_name])
    avg = np.mean(scores[fn_name])
    scores[fn_name] = list(map(lambda x: (x-avg)/std, scores[fn_name]))

corr = [[] for i in range(len(enroll_sc_fn_names))]

for i in range(len(enroll_sc_fn_names)):
    for j in range(len(enroll_sc_fn_names)):
        Vi = torch.tensor(scores[enroll_sc_fn_names[i]], device=device)
        Vj = torch.tensor(scores[enroll_sc_fn_names[j]], device=device)
        corr[i].append(float(torch.sqrt(torch.dot(Vi-Vj, Vi-Vj)).cpu()))
        print(f"{corr[i][j]:.3f} ", end='')
    print("\n")

state = {'ni-accs': accs["ni"],
         'naswot': accs["naswot"],
         'logsynflow': accs["logsynflow"],
         'ni-times': times["ni"],
         'naswot-times': times["naswot"],
         'logsynflow-times': times["logsynflow"],
         'corr': corr
         }

fname = f"{args.save_loc}/{args.save_string}_{args.nasspace}_{args.dataset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.valid}_{args.test}_{args.seed}.t7"
torch.save(state, fname)
