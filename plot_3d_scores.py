import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Genetic-Based NAS with Hybrid Score Function')
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

filename_ninaswot = f'{args.save_loc}/ninaswot_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_entropy  = f'{args.save_loc}/entropy_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_ntk  = f'{args.save_loc}/ntk_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_acc = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.trainval}.npy'
filename_pickle = f'{args.save_loc}/{args.save_string}_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.pickle'

filenames = {"ninaswot": filename_ninaswot, "entropy": filename_entropy, "ntk": filename_ntk, "acc": filename_acc, "pickle": filename_pickle}

scores = {}
for (k, v) in filenames.items():
    if k == 'pickle': continue
    print(f"{k}: {v}")
    scores[k] = np.load(filenames[k])

mask = np.isnan(scores['ninaswot']) | np.isnan(scores['entropy']) | np.isnan(scores['ntk']) | (scores['ntk'] < -2)# | (scores['acc'] < 88)

for (k, v) in scores.items():
    scores[k] = v[~mask]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('ninaswot')
ax.set_ylabel('entropy')
ax.set_zlabel('nkt')
ax.set_title(f'dataset = {args.dataset}')

color = scores['acc']
cmap = plt.cm.get_cmap('bwr')
norm = plt.Normalize(vmin=0, vmax=100)
ax.scatter(scores['ninaswot'], scores['entropy'], scores['ntk'], c=color, cmap = cmap, norm=norm, s=0.5)

plt.savefig(filenames["pickle"] + '.png')

import pickle
pickle.dump(fig, open(filenames['pickle'], 'wb'))
print(f"result file: {filenames['pickle']}")
