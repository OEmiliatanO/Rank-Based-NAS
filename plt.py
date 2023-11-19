import random
import numpy as np
import matplotlib.pyplot as plt
import statistics
from numpy import std, mean
from scipy.stats import kendalltau
import nasspace
from utils import add_dropout, remap_dataset_names
import datasets
import torch
from Parser import parser
from score import *
import tqdm
from tqdm import trange

args = parser.SA_search_argsparser()
device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
if args.nasspace == "nasbench201":
    args.dataset, acc_type = remap_dataset_names(args.dataset, args.valid, args.test, args.train)
else:
    acc_type = None
print(f"dataset = {args.dataset}, validation = {args.valid}")
searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.valid, args.batch_size, args.augtype, args.repeat, args)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

runs = tqdm.tqdm(total = len(searchspace), desc='progress: ')
scs = {"ni": [], "naswot": [], "acc": []}
n = 0
for uid, net in searchspace:
    if n > 10:
        break
    n += 1
    net = net.to(device)
    ni_sc = ni_score(net, train_loader, device, args)
    scs["ni"].append(ni_sc)
    naswot_sc = naswot_score(net, train_loader, device, args)
    scs["naswot"].append(naswot_sc)
    acc = searchspace.get_final_accuracy(uid, acc_type, args.valid)
    scs["acc"].append(acc)
    runs.update(1)

scs["ni"] = np.array(scs["ni"])
scs["naswot"] = np.array(scs["naswot"])
scs["acc"] = np.array(scs["acc"])

## NASWOT+NI
fig = plt.figure()
ax = fig.add_subplot()
#['ind_rk', 'score_ni', 'score_naswot', 'score_synflow', 'acc', 'rk_tau', 'ni_tau', 'naswot_tau', 'synflow_tau']
ax.set_xlabel(f'acc')
ax.set_ylabel(f'NI + NASWOT')

std_naswot = std(scs['naswot'])
mean_naswot = mean(scs['naswot'])
std_ni = std(scs['ni'])
mean_ni = mean(scs['ni'])

nor_naswot = (scs['naswot'] - mean_naswot) / std_naswot
nor_ni = (scs['ni'] - mean_ni) / std_ni

#mask = np.isfinite(scs['score_synflow']).astype(bool) & (dict['score_synflow'] < 1e38).astype(bool)
tau, p = kendalltau(nor_naswot+nor_ni, scs['acc'])
ax.set_title(f'nasbench201 cifar100 \n $\\tau=${tau:.3f}')

ax.scatter(scs['acc'],nor_naswot+nor_ni,s=0.8)

fig.savefig("ninaswot-acc.pdf")

## NI
fig2 = plt.figure()
ax = fig2.add_subplot()
#['ind_rk', 'score_ni', 'score_naswot', 'score_synflow', 'acc', 'rk_tau', 'ni_tau', 'naswot_tau', 'synflow_tau']
ax.set_xlabel(f'acc')
ax.set_ylabel(f'NI')

#mask = np.isfinite(scs['score_synflow']).astype(bool) & (dict['score_synflow'] < 1e38).astype(bool)
tau, p = kendalltau(nor_ni, scs['acc'])
ax.set_title(f'nasbench201 cifar100 \n $\\tau=${tau:.3f}')

ax.scatter(scs['acc'],nor_ni,s=0.8)

fig2.savefig("ni-acc.pdf")

## NASWOT
fig3 = plt.figure()
ax = fig3.add_subplot()
#['ind_rk', 'score_ni', 'score_naswot', 'score_synflow', 'acc', 'rk_tau', 'ni_tau', 'naswot_tau', 'synflow_tau']
ax.set_xlabel(f'acc')
ax.set_ylabel(f'NASWOT')

#mask = np.isfinite(scs['score_synflow']).astype(bool) & (dict['score_synflow'] < 1e38).astype(bool)
tau, p = kendalltau(nor_naswot, scs['acc'])
ax.set_title(f'nasbench201 cifar100 \n $\\tau=${tau:.3f}')

ax.scatter(scs['acc'],nor_naswot,s=0.8)

fig3.savefig("naswot-acc.pdf")
