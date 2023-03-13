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
from scipy.stats import kendalltau
from score import *

parser = argparse.ArgumentParser(description='Genetic-Based NAS with Hybrid Score Functions')
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

parser.add_argument('--valid', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--train', action='store_true')

parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
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
print(f"Current device is {device}")
dataset = 'fake' if 'fake' in args.dataset else args.dataset
args.dataset = args.dataset.replace('fake', '')

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
    assert False, "Unknown dataset: {args.dataset}"

print(f"Initialize the train loader.")
train_loader = datasets.get_data(args.dataset, args.data_loc, args.valid, args.batch_size, args.repeat, args)

print(f"Initialize the nasspace.")
args.dataset, acc_type = remap_dataset_names(args.dataset, args.valid, args.test, args.train)
print(f"dataset = {args.dataset}, valid = {args.valid}, test = {args.test}")
searchspace = nasspace.get_search_space(args)

os.makedirs(args.save_loc, exist_ok=True)

filename_ninaswot = f'{args.save_loc}/ninaswot_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}'
#filename_entropy  = f'{args.save_loc}/entropy_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}'
filename_ntk      = f'{args.save_loc}/ntk_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}'
filename_gradsign = f'{args.save_loc}/gradsign_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}'
filename_acc = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.valid}'

filenames = {"ninaswot": filename_ninaswot, "ntk": filename_ntk, "acc": filename_acc}
print(f"files to save: {filenames}")

scores = dict(zip(["ninaswot", "ntk", "gradsign"],[np.zeros(len(searchspace)) for i in range(3)]))
accs = np.zeros(len(searchspace))

print(f"Start calculate means and stds in {args.n_samples} samples")
means, stds = get_mean_std(searchspace, args.n_samples, train_loader, device, args)
means["ninaswot"] = 0
stds["ninaswot"]  = np.sqrt(5)
print(f"Done")
print(f"means = {means}")
print(f"stds  = {stds}")

print(f"Now score the whole arches.")
nruns = tqdm(total = len(searchspace))
times = []
for i, (uid, network) in enumerate(searchspace):
    st = time.time()
    try:
        #standardize = lambda x, m, s: (x-m)/s
        
        # ninaswot
        # ninaswot has mean 0, and std sqrt5 (naswot*2+ni)
        scores['ninaswot'][uid] = standardize(ninaswot_score(network, train_loader, device, stds, means, args), means["ninaswot"], stds["ninaswot"])

        # ntk
        scores['ntk'][uid] = -standardize(ntk_score(network, train_loader, device), means["ntk"], stds["ntk"])

        # entropy
        #network = init_net_gaussian(network, device)
        #scores['entropy'][uid] = standardize(entropy_score(network, train_loader, device, args), means["entropy"], stds["entropy"])

        # gradsign
        #scores['gradsign'][uid] = standardize(gradsign_score(network, train_loader, device), means["gradsign"], stds["gradsign"])

        accs[uid] = searchspace.get_final_accuracy(uid, acc_type, args.valid)
        if i % 1000 == 0:
            pass
            np.save(filenames['ninaswot'], scores['ninaswot'])
            np.save(filenames['ntk'], scores['ntk'])
            #np.save(filenames['entropy'], scores['entropy'])
            #np.save(filenames['gradsign'], scores['gradsign'])
            np.save(filenames['acc'], accs)
    except Exception as e:
        print(e)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.valid)
        break
    times.append(time.time()-st)
    ninaswot_tau, _ = kendalltau(accs[:uid+1], scores["ninaswot"][:uid+1])
    ntk_tau, _      = kendalltau(accs[:uid+1], scores["ntk"][:uid+1])
    maxacc_ninaswot = accs[np.argmax(scores["ninaswot"][:uid+1])]
    maxacc_ntk      = accs[np.argmax(scores["ntk"][:uid+1])]
    #gradsign_tau, _ = kendalltau(accs[:uid+1], scores["gradsign"][:uid+1])
    nruns.set_description(f"average elapse={mean(times):.2f}, ninaswot tau={ninaswot_tau:.3f}, ntk tau={ntk_tau:.3f}, maxacc(ninaswot)={maxacc_ninaswot}, maxacc(ntk)={maxacc_ntk}")
    nruns.update(1)

np.save(filenames['ninaswot'], scores['ninaswot'])
np.save(filenames['ntk'], scores['ntk'])
#np.save(filenames['entropy'], scores['entropy'])
#np.save(filenames['gradsign'], scores['gradsign'])
np.save(filenames['acc'], accs)
