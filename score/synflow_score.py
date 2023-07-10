import torch
import torch.nn as nn
import numpy as np
import copy

def synflow_score(network, train_loader, device):
    @torch.no_grad()
    def linearize(network):
        signs = {}
        for name, param in network.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(network, signs):
        for name, param in network.state_dict().items():
            param.mul_(signs[name])
    
    #network = copy.deepcopy(network)
    # disable BN layer and dropout
    #network.eval()

    network = network.to(device)
    signs = linearize(network)

    data, _ = next(iter(train_loader))
    input_dim = list(data[0,:].shape)
    input = torch.ones([1]+input_dim).to(device)
    output, logit = network(input)
    torch.sum(output).backward()

    grads = []
    
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight.grad * layer.weight)
        else:
            return torch.zeros_like(layer.weight)

    for layer in network.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm2d):
            grads.append(synflow(layer))

    score = 0
    for i in range(len(grads)):
        score += torch.sum(grads[i])
    
    torch.cuda.empty_cache()
    return score.detach().cpu().numpy()
