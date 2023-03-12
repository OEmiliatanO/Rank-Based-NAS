import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def get_flattened_metric(net, metric):
    grad_list = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grad_list.append(metric(layer).flatten())
    flattened_grad = np.concatenate(grad_list)
    return flattened_grad

def get_grad_conflict(net, inputs, targets, loss_fn=F.cross_entropy):
    N = inputs.shape[0]
    batch_grad = []
    for i in range(N):
        net.zero_grad()
        outputs, logit = net.forward(inputs[[i]]) # nasbence201 return outputs and logit
        #print(f"type(outputs)={type(outputs)}")
        loss = loss_fn(logit, targets[[i]])
        #print(f"type(loss)={type(loss)}")
        loss.backward()
        flattened_grad = get_flattened_metric(net, lambda
            l: l.weight.grad.data.cpu().numpy() if l.weight.grad is not None else torch.zeros_like(l.weight).numpy())
        batch_grad.append(flattened_grad)
    batch_grad = np.stack(batch_grad)
    direction_code = np.sign(batch_grad)
    direction_code = abs(direction_code.sum(axis=0))
    score = np.nanmean(direction_code)
    return score

def gradsign_score(network, train_loader, device):
    data_iterator = iter(train_loader)
    x, target = next(data_iterator)
    x, target = x.to(device), target.to(device)
    return get_grad_conflict(network, x, target)
