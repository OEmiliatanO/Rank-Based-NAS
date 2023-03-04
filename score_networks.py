import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
import time
from tqdm import tqdm, trange
from torch import nn
from scipy import stats
from pycls.models.nas.nas import Cell
from utils import add_dropout, init_network
from statistics import mean
from score import *

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--n_samples', default=15, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")
savedataset = args.dataset
dataset = 'fake' if 'fake' in args.dataset else args.dataset
args.dataset = args.dataset.replace('fake', '')
if args.dataset == 'cifar10':
    args.dataset = args.dataset + '-valid'
searchspace = nasspace.get_search_space(args)
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)

filename_ninaswot = f'{args.save_loc}/ninaswot_logdet_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}'
filename_entropy  = f'{args.save_loc}/entropy_mean_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}'
filename_ntk  = f'{args.save_loc}/ntk_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}'
accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.trainval}'

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'


scores_ninaswot = np.zeros(len(searchspace))
scores_entropy  = np.zeros(len(searchspace))
scores_ntk      = np.zeros(len(searchspace))
scores          = np.zeros(len(searchspace))

try:
    accs = np.load(accfilename + '.npy')
except:
    accs = np.zeros(len(searchspace))

print(f"Start calculate means and stds in {args.n_samples} samples")
means, stds = get_mean_std(searchspace, 15, train_loader, device, args)
means["ninaswot"] = 0
stds["ninaswot"]  = np.sqrt(5)
print(f"Done")
print(f"means = {means}")
print(f"stds  = {stds}")

nruns = tqdm(total = len(searchspace))
times = []
print(f"Now score the whole arches")
for i, (uid, network) in enumerate(searchspace):
    st = time.time()
    try:
        #standardize = lambda x, m, s: (x-m)/s
        
        # ninaswot
        # ninaswot has mean 0, and std sqrt5 (naswot*2+ni)
        scores_ninaswot[i] = standardize(ninaswot_score(network, train_loader, device, stds, means, args), means["ninaswot"], stds["ninaswot"])

        # ntk
        scores_ntk[i] = -standardize(ntk_score(network, train_loader, device, train_mode=args.trainval), means["ntk"], stds["ntk"])

        # entropy
        network = init_net_gaussian(network, device)
        scores_entropy[i] = standardize(entropy_score(network, train_loader, device, args), means["entropy"], stds["entropy"])

        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        if i % 1000 == 0:
            np.save(filename_ninaswot, scores_ninaswot)
            np.save(filename_ntk, scores_ntk)
            np.save(filename_entropy, scores_entropy)
            np.save(accfilename, accs)
    except Exception as e:
        print(e)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        break
    times.append(time.time()-st)
    nruns.set_description(f"average elapse={mean(times):.2f}")
    nruns.update(1)

np.save(filename_ninaswot, scores_ninaswot)
np.save(filename_ntk, scores_ntk)
np.save(filename_entropy, scores_entropy)
np.save(accfilename, accs)
