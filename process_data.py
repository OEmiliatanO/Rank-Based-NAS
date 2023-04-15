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

parser.add_argument('--valid', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--train', action='store_true')

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
parser.add_argument('--find_the_problem', action='store_true')
parser.add_argument('--plot', action='store_true')

args = parser.parse_args()

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

args.dataset, acc_type = remap_dataset_names(args.dataset, args.valid, args.test, args.train)

filename_ninaswot = f'{args.save_loc}/ninaswot_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_ntk = f'{args.save_loc}/ntk_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_logsynflow = f'{args.save_loc}/logsynflow_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
filename_acc = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.valid}.npy'
filename_result = f'{args.save_loc}/{args.oper}_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'

filenames = {"acc": filename_acc, "ninaswot":filename_ninaswot, "ntk":filename_ntk, "logsynflow": filename_logsynflow, "result": filename_result}

for (k, v) in filenames.items():
    print(f"{k}:{v}")

scores = {"ninaswot": np.load(filenames["ninaswot"]), "ntk": np.load(filenames["ntk"]), "logsynflow": np.load(filenames["logsynflow"])}
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
oper = oper.replace("logsynflow", "scores['logsynflow']")

means = {'ntk': 8294.156311645507, 'ninaswot': 0, 'logsynflow': 429725.84}
stds = {'ntk': 21499.301880380423, 'ninaswot': 2.23606797749979, 'logsynflow': 353190.9219404517}

print("result = "+oper)

#result = eval(oper)
sumofstds = 1/stds['ntk']+1/stds['ninaswot']+1/stds['logsynflow']
#w = [1/stds['ninaswot']/sumofstds, 1/stds['ntk']/sumofstds, 1/stds['logsynflow']/sumofstds]
w = [1, 1, 0.1]
result = w[0] * scores['ninaswot'] + w[1] * scores['ntk'] + w[2] * scores['logsynflow']

assert len(result) == 15625, "broken"
np.save(filenames["result"], result)

####
if args.find_the_problem:
    mask = np.full(result.shape, False)
    max_score = np.argmax(accs)
    mask = result > result[max_score]
    the_problems = []
    for i in range(len(mask)):
        if mask[i] == True:
            the_problems.append(i)
    the_problems = np.array(the_problems)
    wheres_the_problems = f'{args.save_loc}/acc-{args.oper}_the_problems_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
    np.save(wheres_the_problems, the_problems)
    print(f"save the problems: {the_problems} as {wheres_the_problems}")
####


mask = np.isinf(result).astype(bool) |\
np.isnan(result).astype(bool) |\
(result > 10).astype(bool) |\
(result < 0).astype(bool)

accs = accs[~mask]
result = result[~mask]

mask = np.full(result.shape, False)
print(f"the maximum found acc according to score is {accs[np.argmax(result)]}")
print(f"and the maximum acc is {np.max(accs)}")
max_score = np.argmax(result)

