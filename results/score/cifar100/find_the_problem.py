import numpy as np
import sys

class arch():
    def __init__(self, no, acc, ninaswot, entropy, ntk, tot):
        self.no = no
        self.acc = acc
        self.ninaswot = ninaswot
        self.entropy = entropy
        self.ntk = ntk
        self.tot = tot

ninaswot = np.load("ninaswot_nasbench201_cifar100_none_0.05_1_True_128_1_1.npy")
entropy  = np.load("entropy_nasbench201_cifar100_none_0.05_1_True_128_1_1.npy")
ntk      = np.load("ntk_nasbench201_cifar100_none_0.05_1_True_128_1_1.npy")
#tot      = ninaswot + entropy + ntk
tot      = ninaswot + ntk
acc      = np.load("all_score_accs_nasbench201_cifar100_True.npy")

#the_problems = np.load("acc-ninaswot_add_ntk_add_entropy_the_problems_nasbench201_cifar10_none_0.05_1_True_128_1_1.npy")
the_problems = np.load("acc-ninaswot_add_ntk_the_problems_nasbench201_cifar100_none_0.05_1_True_128_1_1.npy")

max_acc = np.argmax(acc)
print(f"the max acc is {max_acc}")
#print(f"no.\tacc\tninas\tentropy\tntk\ttot")
print(f"no.\tacc\tninas\tntk\ttot")
#print(f"{max_acc}\t{acc[max_acc]:.3f}\t{ninaswot[max_acc]:.3f}\t{entropy[max_acc]:.3f}\t{ntk[max_acc]:.3f}\t{tot[max_acc]:.3f}")
print(f"{max_acc}\t{acc[max_acc]:.3f}\t{ninaswot[max_acc]:.3f}\t{ntk[max_acc]:.3f}\t{tot[max_acc]:.3f}")
print(f"===========================================================================")
print(f"the problems are here ({len(the_problems)}):")
#print(f"no.\tacc\tninas\tentropy\tntk\ttot")
print(f"no.\tacc\tninas\tntk\ttot")

the_problem_arches = [arch(uid,acc[uid],ninaswot[uid],entropy[uid],ntk[uid],tot[uid]) for uid in the_problems]
the_problem_arches.sort(key = lambda this: this.tot, reverse=True)
mean_acc = 0
for problem in the_problem_arches:
    #print(f"{problem.no}\t{problem.acc:.3f}\t{problem.ninaswot:.3f}\t{problem.entropy:.3f}\t{problem.ntk:.3f}\t{problem.tot:.3f}")
    if problem.acc < 50: continue
    mean_acc += problem.acc
    print(f"{problem.no}\t{problem.acc:.3f}\t{problem.ninaswot:.3f}\t{problem.ntk:.3f}\t{problem.tot:.3f}")
print(f"mean acc: {mean_acc/44}")
"""
for the_problem in the_problem_arch:
    #if ninaswot[the_problem] < ninaswot[max_acc]:continue
    #if entropy[the_problem] < entropy[max_acc]:continue
    #if ntk[the_problem] < ntk[max_acc]:continue
    print(f"{the_problem}\t{acc[the_problem]:.3f}\t{ninaswot[the_problem]:.3f}\t{entropy[the_problem]:.3f}\t{ntk[the_problem]:.3f}\t{tot[the_problem]:.3f}")
"""

arches = []
for i in range(len(acc)):
    arches.append(arch(i, acc[i], ninaswot[i], entropy[i], ntk[i], tot[i]))

arches.sort(key = lambda this: this.tot)
print("================================================")
print(f"sort by tot")
print(f"no.\tacc\tninas\tentropy\tntk\ttot")
for this in arches:
    print(f"{this.no}\t{this.acc:.3f}\t{this.ninaswot:>.3f}\t{this.entropy:>.3f}\t{this.ntk:>.3f}\t{this.tot:>.3f}")


def sort_and_prune(arches, what, proposition):
    arches.sort(key = lambda this: eval(f"this.{what}"), reverse = True)
    print("================================================")
    print(f"sort by {what} and prune {proposition}")
    arches = arches[:int(len(arches)*proposition)]
    print(f"no.\tacc\tninas\tentropy\tntk\ttot")
    for this in arches:
        print(f"{this.no}\t{this.acc:.3f}\t{this.ninaswot:>.3f}\t{this.entropy:>.3f}\t{this.ntk:>.3f}\t{this.tot:>.3f}")
    return arches

#arches = sort_and_prune(arches, "ninaswot", 0.2)
#arches = sort_and_prune(arches, "entropy", 0.6)
#arches = sort_and_prune(arches, "ntk", 0.5)
#arches = sort_and_prune(arches, "ninaswot", 0.2)

#arches = sort_and_prune(arches, "ninaswot", 0.2)
#arches = sort_and_prune(arches, "entropy", 0.6)
#arches = sort_and_prune(arches, "ntk", 0.5)
#arches = sort_and_prune(arches, "ninaswot", 0.2)
