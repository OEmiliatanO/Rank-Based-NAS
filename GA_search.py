import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from statistics import mean, stdev
import time
from utils import add_dropout
from score import *
import sys
from Parser import parser

args = parser.GAsearch_argsparser()

if args.search_algo == "ori" or args.search_algo == "original" or args.search_algo == "origin" or args.search_algo == "base":
    print(f"Use search algorithm: GA origin ver.")
    from search.GA import GA
elif args.search_algo == "mm":
    print(f"Use search algorithm: GA multiple metric ver.")
    from search.GA_mm import GA
elif args.search_algo == "rk" or args.search_algo == "based_on_rank":
    print(f"Use search algorithm: GA rank-based ver.")
    from search.GA_based_on_rank import GA
else:
    assert False, f"no such search algorithm: {args.GA}"

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print(f"Use GPU {args.GPU}...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The current device used is {device}")

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

times     = []
chosen    = []
acc       = []
topscores = []
naswot_acc = []
ni_acc = []
logsynflow_acc = []
ninaswot_acc = []


if args.search_algo != 'rk':
    print(f"Currently calculate means and standards.")
    means, stds = get_mean_std(searchspace, args.n_samples, train_loader, device, args)
    print(f"Calculation of means and stds is done.")
    print(f"means = {means}\nstds = {stds}")
    print(f"========================================")

means, stds = {}, {}

print(f"parameter:\nnumber of population={args.maxn_pop}\nnumber of iteration={args.maxn_iter}\nprobability of mutation={args.prob_mut}\nprobability of crossover={args.prob_cr}")

print(f"========================================")
runs = trange(args.n_runs, desc='Unavailable')
for N in runs:
    start = time.time()

    # nas-bench-201 spec
    sol = GA(6, 5, searchspace, train_loader, device, stds, means, acc_type, args)
    if args.search_algo == "rk":
        if args.verbose:
            score, acc_, uid, rk, naswotacc, niacc, logsynflowacc, ninaswotacc = sol.find_best()
            naswot_acc.append(naswotacc)
            ni_acc.append(niacc)
            logsynflow_acc.append(logsynflowacc)
            ninaswot_acc.append(ninaswotacc)
        else:
            score, acc_, uid, rk = sol.find_best()
    else:
        score, acc_, uid = sol.find_best()
    chosen.append(uid)
    topscores.append(score)
    acc.append(acc_)

    times.append(time.time()-start)
    if isinstance(topscores[-1], tuple):
        if args.verbose:
            runs.set_description(f"rk-acc: {mean(acc):.3f}%({(stdev(acc) if len(acc) > 1 else 0):.3f}), naswot-acc: {mean(naswot_acc):.3f}%({(stdev(naswot_acc) if len(naswot_acc) > 1 else 0):.3f}), ni-acc: {mean(ni_acc):.3f}%({(stdev(ni_acc) if len(ni_acc) > 1 else 0):.3f}), logsyn-acc: {mean(logsynflow_acc):.3f}%({(stdev(logsynflow_acc) if len(logsynflow_acc) > 1 else 0):.3f}), ninaswot-acc: {mean(ninaswot_acc):.3f}%({(stdev(ninaswot_acc) if len(ninaswot_acc) > 1 else 0):.3f}), time:{mean(times):.2f}")
        else:
            runs.set_description(f"rk-acc: {mean(acc):.3f}%({(stdev(acc) if len(acc) > 1 else 0):.3f}) time:{mean(times):.2f}")

    else:
        runs.set_description(f"acc: {mean(acc):.3f}%  acc std: {(stdev(acc) if len(acc) > 1 else 0):.3f} time:{mean(times):.2f}")

print(f"Final mean accuracy: {np.mean(acc)}")

state = {'accs': acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

fname = f"{args.save_loc}/{args.save_string}_{args.nasspace}_{args.dataset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}_{args.valid}_{args.test}_{args.maxn_pop}_{args.maxn_iter}_{args.prob_cr}_{args.prob_mut}.t7"
torch.save(state, fname)
