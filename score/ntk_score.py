import numpy as np
import torch
import time
import copy

def ntk_score(network, train_loader, device, args, recalbn=0, train_mode=True, num_batch=1):
    ntk = []
    network = copy.deepcopy(network)
    if train_mode:
        network.train()
    else:
        network.eval()
    ######
    network.to(device)
    grads = []
    for i, (x, targets) in enumerate(train_loader):
        if num_batch > 0 and i >= num_batch: break
        x = x.to(device)
            
        network.zero_grad()
        x_ = x.clone().to(device)
        logit = network(x_)
        if isinstance(logit, tuple):
            logit = logit[1]  # 201 networks: return features and logits
        for _idx in range(len(x_)):
            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            grad = []
            for name, W in network.named_parameters():
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
            grads.append(torch.cat(grad, -1))
            network.zero_grad()
            #torch.cuda.empty_cache()
    ######
    grads = torch.stack(grads, 0)
    ntk = torch.einsum('nc,mc->nm', [grads, grads])
    conds = []
    eigenvalues, eigenvectors = torch.symeig(ntk, upper=True)  # ascending
    score = np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)
    del network
    return float(score)
