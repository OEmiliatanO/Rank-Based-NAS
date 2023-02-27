import numpy as np
import torch
from torch import nn

def init_net_gu(net, device):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(k1 * k2 * in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net.to(device)

def entropy_score(network, train_loader, device):
    network.features = []
    def forward_hook(module, inp, out):
        try:
            if not module.visitied_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            #print(f"inp.size() = {inp.size()}")
            #for i in range(inp.size()[0]):
            #    for j in range(inp.size()[1]):
            #        network.features.append(torch.std(inp[i][j]))
            #print(f"torch.mean(inp) = {torch.mean(inp)}")
            #print(f"torch.std(inp)  = {torch.std(inp)}")
            network.features.append(torch.sum(torch.abs(inp), dim = [1,2,3]))
            #print(f"network.features[-1].size() = {network.features[-1].size()}")
        except:
            pass

    def backward_hook(module, inp, out):
        module.visitied_backwards = True

    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    network = network.to(device)
    data_iter = iter(train_loader)
    x, target = next(data_iter)
    x, target = x.to(device), target.to(device)

    noise = x.new(x.size()).normal_(0, 1).to(device)
    #print(f"x.shape = {x.shape}")
    jacobs, labels, y, out = get_batch_jacobian(network, noise, target, device, args)
    out, _ = network(noise)
    out = out.to(device).detach()
    scores = 0
    for i in range(len(network.features)):
        scores += torch.log(torch.mean(network.features[i]))
        #scores += torch.log(torch.std(network.features[i]))
        #print(f"scores = {scores}")
    del network.features
    print(f"scores = {scores}")
    return scores
