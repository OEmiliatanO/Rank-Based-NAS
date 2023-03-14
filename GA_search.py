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
from search.GA_based_on_rank import GA
#from search.GA import GA
from score import *


parser = argparse.ArgumentParser(description='Genetic-Based NAS algorithm with Hybrid Score Functions')

parser.add_argument('--maxn_pop', default=25, type=int, help='number of population')
parser.add_argument('--maxn_iter', default=30, type=int, help='number of iteration')
parser.add_argument('--prob_mut', default=0.07, type=float, help='probability of mutation')
parser.add_argument('--prob_cr', default=0.8, type = float, help='probability of crossover')

parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201.pth',
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
parser.add_argument('--n_samples', default=50, type=int)
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
train_loader = datasets.get_data(args.dataset, args.data_loc, args.valid, args.batch_size, args.repeat, args)

print(f"Initialize the nas bench api...")
args.dataset, acc_type = remap_dataset_names(args.dataset, args.valid, args.test, args.train)
print(f"dataset = {args.dataset}, validation = {args.valid}")
searchspace = nasspace.get_search_space(args)

print(f"Make sure {args.save_loc} exist.")
os.makedirs(args.save_loc, exist_ok=True)

times     = []
chosen    = []
acc       = []
topscores = []

print(f"Now calculate means and standards.")
#means, stds = get_mean_std(searchspace, args.n_samples, train_loader, device, args)

if args.dataset == 'cifar10-valid':
    # cifar10-valid
    means = {}
    stds = {}
    means["ni"]       =    -0.08287504196166992
    stds["ni"]        =     0.1338795386377707
    means["naswot"]   =  1595.1744876295015
    stds["naswot"]    =    76.14891987423042
    means["ntk"]      = 10149.060118942261
    stds["ntk"]       = 28253.425318545123
    means["ninaswot"] = 0
    stds["ninaswot"]  = np.sqrt(5)
elif args.dataset == 'cifar100':
    # cifar100
    means, stds = get_mean_std(searchspace, args.n_samples, train_loader, device, args)
    """
    means = {}
    stds = {}
    means["ni"] = -0.07765040397644044
    stds["ni"]  =  0.12295018402244674
    means["naswot"] = 1592.1985007236365
    stds["naswot"] = 77.2502068400697
    means["ntk"] = 9979.810939865112
    stds["ntk"] = 23336.402907292802
    means["ninaswot"] = 0
    stds["ninaswot"]  = np.sqrt(5)
    """
else:
    means, stds = get_mean_std(searchspace, args.n_samples, train_loader, device, args)


print(f"Calculation is done.")
print(f"means = {means}\nstds = {stds}")
print(f"========================================")

print(f"parameter:\nnumber of population={args.maxn_pop}\nnumber of iteration={args.maxn_iter}\nprobability of mutation={args.prob_mut}\nprobability of crossover={args.prob_cr}")

print(f"========================================")
runs = trange(args.n_runs, desc='acc: nan topscores: nan')
for N in runs:
    start = time.time()

    # nas-bench-201 spec
    sol = GA(6, 5, searchspace, train_loader, device, stds, means, acc_type, args)
    score, acc_, uid = sol.find_best()
    chosen.append(uid)
    topscores.append(score)
    acc.append(acc_)

    times.append(time.time()-start)
    runs.set_description(f"acc: {mean(acc):.3f}%  acc std: {(stdev(acc) if len(acc) > 1 else 0):.3f}  topscores:({topscores[-1][0]:.3f},{topscores[-1][1]:.3f},{topscores[-1][2]:.3f})  time:{mean(times):.2f}")
    #runs.set_description(f"acc: {mean(acc):.3f}%  acc std: {(stdev(acc) if len(acc) > 1 else 0):.3f}  uid: {uid}  topscores:{topscores[-1]:.3f}  time:{mean(times):.2f}")

print(f"Final mean accuracy: {np.mean(acc)}")

state = {'accs': acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

fname = f"{args.save_loc}/{args.save_string}_{args.nasspace}_{args.dataset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}_{args.valid}_{args.test}_{args.maxn_pop}_{args.maxn_iter}_{args.prob_cr}_{args.prob_mut}.t7"
torch.save(state, fname)
