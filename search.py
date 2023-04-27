import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from statistics import mean
from numpy import std
import time
from utils import add_dropout
from score import *
from encoder import encoder

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='./NAS-Bench-201.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results/ICML', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--kernel', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)

parser.add_argument('--valid', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--train', action='store_true')

parser.add_argument('--activations', action='store_true')
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
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

print(f"Initialize the nas bench api...")
args.dataset, acc_type = remap_dataset_names(args.dataset, args.valid, args.test, args.train)
print(f"dataset = {args.dataset}, validation = {args.valid}")
searchspace = nasspace.get_search_space(args)

print(f"Making sure {args.save_loc} exist.")
os.makedirs(args.save_loc, exist_ok=True)

"""
print(f"Currently calculate means and standards.")
means, stds = get_mean_std(searchspace, 100, train_loader, device, args)
print(f"Calculation of means and stds is done.")
print(f"means = {means}\nstds = {stds}")
"""
print(f"========================================")

times = []
accs = {"ni": [], "naswot": [], "logsynflow": [], "rank-based": []}

Encoder = encoder.get_encoder(args.nasspace)

runs = trange(args.n_runs, desc='acc: ')
for N in runs:
    start = time.time()
    #indices = np.random.randint(0,len(searchspace),args.n_samples)
    codebase = Encoder.get_nrand_code(args.n_samples)
    indices = np.array([searchspace.get_index_by_code(Encoder.parse_code(c)) for c in codebase])

    scores = {"ni": [], "naswot": [], "logsynflow": []}
    
    for uid in indices:
        network = searchspace.get_network(uid)
        network = network.to(device)
        scores["ni"].append(ni_score(network, train_loader, device, args))
        scores["naswot"].append(naswot_score(network, train_loader, device, args))
        scores["logsynflow"].append(logsynflow_score(network, train_loader, device))
        del network

    totrk = dict(zip([uid for uid in indices], [0 for i in range(args.n_samples)]))
    
    rk_ni = indices[np.argsort(scores["ni"])]
    for rk, id in enumerate(rk_ni):
        totrk[id] += args.n_samples - rk
    
    rk_naswot = indices[np.argsort(scores["naswot"])]
    for rk, id in enumerate(rk_naswot):
        totrk[id] += args.n_samples - rk

    rk_logsynflow = indices[np.argsort(scores["logsynflow"])]
    bestrk = np.inf
    for rk, id in enumerate(rk_logsynflow):
        totrk[id] += args.n_samples - rk

        if bestrk > totrk[id]:
            bestrk = totrk[id]
            bestrk_uid = id

    accs["ni"].append(searchspace.get_final_accuracy(rk_ni[-1], acc_type, args.valid))
    accs["naswot"].append(searchspace.get_final_accuracy(rk_naswot[-1], acc_type, args.valid))
    accs["logsynflow"].append(searchspace.get_final_accuracy(rk_logsynflow[-1], acc_type, args.valid))
    accs["rank-based"].append(searchspace.get_final_accuracy(bestrk_uid, acc_type, args.valid))

    times.append(time.time()-start)
    runs.set_description(f"acc_ni: {mean(accs['ni']):.2f}({std(accs['ni']):.3f}), acc_naswot: {mean(accs['naswot']):.2f}({std(accs['naswot']):.3f}), acc_logsyn: {mean(accs['logsynflow']):.2f}({std(accs['logsynflow']):.3f}), rk-based: {mean(accs['rank-based']):.2f}({std(accs['rank-based']):.3f})")

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
