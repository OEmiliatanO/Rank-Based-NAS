import torch
import torch.nn as nn
import numpy as np

def logsynflow_score(network, train_loader, device):
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
    
    # disable BN layer and dropout
    network.eval()

    network = network.to(device)
    signs = linearize(network)

    data, _ = next(iter(train_loader))
    input_dim = list(data[0,:].shape)
    input = torch.ones([1]+input_dim).to(device)
    output, logit = network(input)
    torch.sum(output).backward()

    grads_abs = []
    
    def synflow(layer):
        if layer.weight.grad is not None:
            res = torch.abs(torch.log(layer.weight.grad)) * layer.weight
            res = res[res.isfinite()]
            return res
        else:
            return torch.zeros_like(layer.weight)

    score = 0
    for layer in network.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            score += torch.sum(synflow(layer))

    """
    for p in parameters():
        scores[id(p)] = torch.clone(np.log(p.grad) * p).detach().abs_().cpu().numpy()
        p.grad.data.zero_()
    """

    nonlinearize(network, signs)
    return score.detach().cpu().numpy()
