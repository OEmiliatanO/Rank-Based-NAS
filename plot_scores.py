import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib
from decimal import Decimal
from scipy.special import logit, expit
from scipy import stats
import seaborn as sns

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--init', default='', type=str)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

filename_ninaswot = f'{args.save_loc}/ninaswot_logdet_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_entropy  = f'{args.save_loc}/entropy_mean_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_ntk  = f'{args.save_loc}/ntk_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.trainval}.npy'
filename_pickle = f'{args.save_loc}/{args.save_string}_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.pickle'

print(filename_ninaswot)
print(filename_entropy)
print(filename_ntk)
print(accfilename)
print(filename_pickle)

scores_ninaswot = np.load(filename_ninaswot)
scores_entropy  = np.load(filename_entropy)
scores_ntk      = np.load(filename_ntk)
accs            = np.load(accfilename)

mask = np.isnan(scores_ninaswot) | np.isnan(scores_entropy) | np.isnan(scores_ntk) #| (scores_ntk > 10) | (accs < 88)
print(mask)

scores_ninaswot = scores_ninaswot[~mask]
scores_entropy  = scores_entropy[~mask]
scores_ntk      = scores_ntk[~mask]
accs            = accs[~mask]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('ninaswot')
ax.set_ylabel('entropy')
ax.set_zlabel('nkt')
ax.set_title(f'dataset = {args.dataset}')

color = accs
cmap = plt.cm.get_cmap('bwr')
norm = plt.Normalize(vmin=0, vmax=100)
ax.scatter(scores_ninaswot, scores_entropy, scores_ntk, c=color, cmap = cmap, norm=norm)

plt.savefig(filename_pickle + '.pdf')

import pickle
pickle.dump(fig, open(filename_pickle, 'wb'))
