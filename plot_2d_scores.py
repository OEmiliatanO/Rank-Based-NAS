import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

parser = argparse.ArgumentParser(description='Genetic-Based NAS with Hybrid Score Functions')
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

parser.add_argument('--targets', default="ninaswot-acc")
parser.add_argument('--find_the_problem', action='store_true')

args = parser.parse_args()

def remap_dataset_names(dataset, valid, test, train):
    cifar10 = 'cifar10'
    if dataset == cifar10 and valid:
        return cifar10 + '-valid', 'x-valid'
    if dataset == cifar10 and test:
        return cifar10, 'ori-test'
    if dataset == cifar10 and train:
        return cifar10, 'train'

    assert not train, "train label"
    cifar100 = 'cifar100'
    if dataset == cifar100 and valid:
        return cifar100, 'x-valid'
    if dataset == cifar100 and test:
        return cifar100, 'x-test'
    
    ImageNet16_120 = "ImageNet16-120"
    if dataset == ImageNet16_120 and valid:
        return ImageNet16_120, 'x-valid'
    if dataset == ImageNet16_120 and test:
        return ImageNet16_120, 'x-test'
    assert False, "Unknown dataset: {args.dataset}"

args.dataset, acc_type = remap_dataset_names(args.dataset, args.valid, args.test, args.train)

targets = args.targets
targets = targets.split('-')
assert len(targets) == 2, "length of targets must be 2"
print(targets[0])
print(targets[1])

filenames = [f'{args.save_loc}/{targets[i]}_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}.npy' for i in range(len(targets))]

for i, fname in enumerate(filenames):
    if "acc" in fname:
        filenames[i] = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.valid}.npy'
        break

for fname in filenames:
    print(fname)

scores = [np.load(filenames[i]) for i in range(len(filenames))]

assert len(scores[0]) == 15625, "the length of scores isn't correct"

####
if args.find_the_problem:
    mask = np.full(scores[0].shape, False)
    for i, fname in enumerate(filenames):
        if "acc" in fname:
            max_score = np.argmax(scores[i])
            mask = scores[1-i] > scores[1-i][max_score]
    the_problems = []
    for i in range(len(mask)):
        if mask[i] == True:
            the_problems.append(i)
    the_problems = np.array(the_problems)
    wheres_the_problems = f'{args.save_loc}/acc-{targets[1]}_the_problems_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
    np.save(wheres_the_problems, the_problems)
    print(f"save the problems: {the_problems}")
####

mask = np.isinf(scores[1]).astype(bool) |\
np.isnan(scores[1]).astype(bool) |\
(scores[1] > 10).astype(bool) |\
(scores[1] < 0).astype(bool) |\
(scores[1] < 1).astype(bool)

scores[0] = scores[0][~mask]
scores[1] = scores[1][~mask]

mask = np.full(scores[0].shape, False)
for i, fname in enumerate(filenames):
    if "acc" in fname:
        print(f"the maximum found acc according to score is {scores[i][np.argmax(scores[1-i])]}")
        print(f"and the maximum acc is {np.max(scores[i])}")
        max_score = np.argmax(scores[i])
        if args.find_the_problem:
            mask = scores[1-i] > scores[1-i][max_score]

if args.find_the_problem:
    scores[0] = scores[0][mask]
    scores[1] = scores[1][mask]

tau, p = stats.kendalltau(scores[0], scores[1])

fig = plt.figure()
ax = fig.add_subplot()

ax.set_xlabel(f'{targets[0]}')
ax.set_ylabel(f'{targets[1]}')
ax.set_title(f'{args.nasspace} {args.dataset} \n $\\tau=${tau:.3f}')

filename_result = f'{args.save_loc}/{args.save_string}_{args.targets}_{args.nasspace}_{args.dataset}_{args.augtype}_{args.sigma}_{args.repeat}_{args.valid}_{args.batch_size}_{args.maxofn}_{args.seed}'

print("result file :" + filename_result + ".png")
print("result file :" + filename_result + ".pdf")

ax.scatter(x=scores[0], y=scores[1], s = 0.3)
plt.savefig(filename_result + ".png")
plt.savefig(filename_result + ".pdf")
