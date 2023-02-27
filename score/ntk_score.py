import numpy as np
import torch

def ntk_score(xloader, network, recalbn=0, train_mode=False, num_batch=-1):
    device = torch.cuda.current_device()
    ntk = []
    if train_mode:
        network.train()
    else:
        network.eval()
    ######
    grads = []
    for i, (inputs, targets) in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)
            
        network.zero_grad()
        inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
        logit = network(inputs_)
        if isinstance(logit, tuple):
            logit = logit[1]  # 201 networks: return features and logits
        for _idx in range(len(inputs_)):
            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            grad = []
            for name, W in network.named_parameters():
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
            grads.append(torch.cat(grad, -1))
            network.zero_grad()
            torch.cuda.empty_cache()
    ######
    grads = torch.stack(grads, 0)
    ntk = torch.einsum('nc,mc->nm', [grads, grads])
    conds = []
    eigenvalues, _ = torch.symeig(ntk)  # ascending
    score = np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)
    return score
