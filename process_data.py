import argparse
import numpy as np
from scipy import stats

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

parser.add_argument('--oper', default='ninaswot_add_ntk_add_entropy',type=str)

args = parser.parse_args()

filename_ninaswot = f'{args.save_loc}/ninaswot_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_ntk = f'{args.save_loc}/ntk_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_entropy = f'{args.save_loc}/entropy_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_acc = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.trainval}.npy'
filename_result = f'{args.save_loc}/{args.oper}_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'

filenames = {"acc": filename_acc, "ninaswot":filename_ninaswot, "ntk":filename_ntk, "entropy": filename_entropy, "result": filename_result}

for (k, v) in filenames.items():
    print(f"{k}:{v}")

scores = {"ninaswot": np.load(filenames["ninaswot"]), "ntk": np.load(filenames["ntk"]), "entropy": np.load(filenames["entropy"])}
accs = np.load(filenames["acc"])

"""
mask = np.full(scores["ninaswot"].shape, False)
for (k, v) in scores.items():
    mask |= ~np.isfinite(v)

for (k, v) in scores.items():
    scores[k] = scores[k][~mask]
"""

oper = args.oper
oper = oper.replace("_add_", "+")
oper = oper.replace("_mul_", "*")
oper = oper.replace("_div_", "/")
oper = oper.replace("_minus_", "-")
oper = oper.replace("ninaswot", "scores['ninaswot']")
oper = oper.replace("ntk", "scores['ntk']")
oper = oper.replace("entropy", "scores['entropy']")

print("result = "+oper)

result = eval(oper)

np.save(filenames["result"], result)
