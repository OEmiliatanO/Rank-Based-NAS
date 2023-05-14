import numpy as np
import sys

class arch():
    def __init__(self, no, acc, ninaswot, ntk, tot):
        self.no = no
        self.acc = acc
        self.ninaswot = ninaswot
        self.ntk = ntk
        self.tot = tot

ninaswot = np.load("ninaswot_nasbench201_cifar10-valid_none_0.05_1_True_128_1_1.npy")
ntk      = np.load("ntk_nasbench201_cifar10-valid_none_0.05_1_True_128_1_1.npy")
logsynflow = np.load("logsynflow_nasbench201_cifar10-valid_none_0.05_1_True_128_1_1.npy")
acc      = np.load("all_score_accs_nasbench201_cifar10-valid_True.npy")
for i in range(len(logsynflow)):
    logsynflow[i] = logsynflow[i] if np.isfinite(logsynflow[i]) else -np.inf
tot      = ninaswot + ntk + logsynflow
ninaswot_rk = np.argsort(ninaswot)
ntk_rk      = np.argsort(ntk)
logsynflow_rk  = np.argsort(logsynflow)
rk_uid  = {}
rk_ninaswot = {}
rk_ntk = {}
rk_logsynflow = {}

for i in range(len(ninaswot_rk)):
    rk_ninaswot[ninaswot_rk[i]] = len(ninaswot_rk) - i
    rk_uid[ninaswot_rk[i]] = rk_ninaswot[ninaswot_rk[i]]
for i in range(len(ntk_rk)):
    rk_ntk[ntk_rk[i]] = len(ntk_rk) - i
    rk_uid[ntk_rk[i]] += rk_ntk[ntk_rk[i]]
for i in range(len(logsynflow_rk)):
    rk_logsynflow[logsynflow_rk[i]] = len(logsynflow_rk) - i
    rk_uid[logsynflow_rk[i]] += rk_logsynflow[logsynflow_rk[i]]

acc_arg = np.argsort(acc)[::-1]
print(f"no.\tacc\tninas\tntk\tlgsyn\ttot\trk")
for uid in acc_arg:
    print(f"{uid}\t{acc[uid]:.3f}\t{ninaswot[uid]:.3f}\t{ntk[uid]:.3f}\t{logsynflow[uid]:.3f}\t{tot[uid]:.3f}")

print(f"based on rank")
print(f"no.\tacc\tninas\tntk\tlogsynflow\ttot\trk")
rk_uid = sorted(rk_uid.items(), key = lambda this: this[1])
for i in range(len(rk_uid)):
    uid = rk_uid[i][0]
    print(f"{uid}\t{acc[uid]}\t{ninaswot[uid]}\t{ntk[uid]}\t{logsynflow[uid]}\t{tot[uid]}\t{rk_uid[i][1]}={rk_ninaswot[uid]}+{rk_ntk[uid]}")

#tot      = ninaswot

#the_problems = np.load("acc-ninaswot_add_ntk_add_entropy_the_problems_nasbench201_cifar10_none_0.05_1_True_128_1_1.npy")
#the_problems = np.load("acc-ninaswot_add_ntk_the_problems_nasbench201_cifar10_none_0.05_1_True_128_1_1.npy")
the_problems = np.load("acc-ninaswot_the_problems_nasbench201_cifar10-valid_none_0.05_1_True_128_1_1.npy")

max_acc = np.argmax(acc)
print(f"the max acc is {max_acc}")
#print(f"no.\tacc\tninas\tentropy\tntk\ttot")
print(f"no.\tacc\tninas\tntk\ttot")
#print(f"{max_acc}\t{acc[max_acc]:.3f}\t{ninaswot[max_acc]:.3f}\t{entropy[max_acc]:.3f}\t{ntk[max_acc]:.3f}\t{tot[max_acc]:.3f}")
print(f"{max_acc}\t{acc[max_acc]:.3f}\t{ninaswot[max_acc]:.3f}\t{ntk[max_acc]:.3f}\t{tot[max_acc]:.3f}")
print(f"===========================================================================")
#print(f"the problems are here:")
#print(f"no.\tacc\tninas\tentropy\tntk\ttot")
print(f"no.\tacc\tninas\tntk\ttot")

the_problem_arches = [arch(uid,acc[uid],ninaswot[uid],ntk[uid],tot[uid]) for uid in the_problems]
the_problem_arches.sort(key = lambda this: this.tot, reverse=True)
for problem in the_problem_arches:
    #print(f"{problem.no}\t{problem.acc:.3f}\t{problem.ninaswot:.3f}\t{problem.entropy:.3f}\t{problem.ntk:.3f}\t{problem.tot:.3f}")
    print(f"{problem.no}\t{problem.acc:.3f}\t{problem.ninaswot:.3f}\t{problem.ntk:.3f}\t{problem.tot:.3f}")
"""
for the_problem in the_problem_arch:
    #if ninaswot[the_problem] < ninaswot[max_acc]:continue
    #if entropy[the_problem] < entropy[max_acc]:continue
    #if ntk[the_problem] < ntk[max_acc]:continue
    print(f"{the_problem}\t{acc[the_problem]:.3f}\t{ninaswot[the_problem]:.3f}\t{entropy[the_problem]:.3f}\t{ntk[the_problem]:.3f}\t{tot[the_problem]:.3f}")
"""

arches = []
for i in range(len(acc)):
    arches.append(arch(i, acc[i], ninaswot[i], ntk[i], tot[i]))



arches.sort(key = lambda this: this.tot)
print("================================================")
print(f"sort by tot")
print(f"no.\tacc\tninas\tentropy\tntk\ttot")
for this in arches:
    print(f"{this.no}\t{this.acc:.3f}\t{this.ninaswot:>.3f}\t{this.ntk:>.3f}\t{this.tot:>.3f}")


def sort_and_prune(arches, what, proposition):
    arches.sort(key = lambda this: eval(f"this.{what}"), reverse = True)
    print("================================================")
    print(f"sort by {what} and prune {proposition}")
    arches = arches[:int(len(arches)*proposition)]
    print(f"no.\tacc\tninas\tntk\ttot")
    for this in arches:
        print(f"{this.no}\t{this.acc:.3f}\t{this.ninaswot:>.3f}\t{this.ntk:>.3f}\t{this.tot:>.3f}")
    return arches

#arches = sort_and_prune(arches, "ninaswot", 0.2)
#arches = sort_and_prune(arches, "entropy", 0.6)
#arches = sort_and_prune(arches, "ntk", 0.5)
#arches = sort_and_prune(arches, "ninaswot", 0.2)

#arches = sort_and_prune(arches, "ninaswot", 0.2)
#arches = sort_and_prune(arches, "entropy", 0.6)
#arches = sort_and_prune(arches, "ntk", 0.5)
#arches = sort_and_prune(arches, "ninaswot", 0.2)

