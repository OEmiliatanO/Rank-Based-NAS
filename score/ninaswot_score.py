import numpy as np
import torch
import random
import copy

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()

def naswot_score(network, train_loader, device, args):
    network = copy.deepcopy(network)
    network.K = np.zeros((args.batch_size, args.batch_size))
    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
        except:
            pass
            
    def counting_backward_hook(module, inp, out):    
        module.visited_backwards = True

    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            if hasattr(module, 'register_full_forward_hook'):
                module.register_full_forward_hook(counting_forward_hook)
                module.register_full_backward_hook(counting_backward_hook)
            else:
                module.register_forward_hook(counting_forward_hook)
                module.register_backward_hook(counting_backward_hook)

    network = network.to(device)    
    s = []

    data_iterator = iter(train_loader)
    x, target = next(data_iterator)
    x2 = torch.clone(x)
    x2 = x2.to(device)
    x, target = x.to(device), target.to(device)
    jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)

    network(x2.to(device))
    s.append(hooklogdet(network.K, target))

    del network
    torch.cuda.empty_cache()
    return np.mean(s)

def ni_extract(network, x, target, device, args):
    network.K = []
    network.n_conv=0
    network.channel = 0
    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            arr = inp.detach().cpu().numpy()
            network.K.append(arr)
            network.channel += arr.shape[1]
        except:
            pass
            
    def counting_backward_hook(module, inp, out):    
        module.visited_backwards = True
    
    def counting_forward_hook_conv(module, inp, out):
        try:
            if not module.visited_backwards_conv:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            arr = inp.detach().cpu().numpy()               
            network.n_conv += 1
            network.channel += arr.shape[1]
        except:
            pass

    def counting_backward_hook_conv(module, inp, out):
        module.visited_backwards_conv = True

    for name, module in network.named_modules():
        if 'Pool' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)
        
        if 'Conv' in str(type(module)):
            module.register_forward_hook(counting_forward_hook_conv)
            module.register_backward_hook(counting_backward_hook_conv)
 
    x2 = torch.clone(x)
    x2 = x2.to(device)
    x, target = x.to(device), target.to(device)
    jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)
            
    network(x2)
    n_conv = network.n_conv
    metric = network.K
    channel = network.channel

    return metric, n_conv, channel

def ni_score(network, train_loader, device, args, debug_no=0, debug_code=""):
    network = network.cuda()
    data_iter = iter(train_loader)
    x, target = next(data_iter)
    x, target = x.to(device), target.float().to(device)
    noise = x.new(x.size()).normal_(0, args.sigma).to(device)
    x2 = x + noise

    data_layers = 0
    n1 = copy.deepcopy(network)
    n1 = n1.to(device)
    target1 = copy.copy(target)
    data_layers, _, _ = ni_extract(n1, x, target1, device, args)

    noise_layers = 0
    n2 = copy.deepcopy(network)
    n2 = n2.to(device)
    target2 = copy.copy(target)
    noise_layers, n_conv, channel = ni_extract(n2, x2, target2, device, args)

    errs = []
    for i in range(len(noise_layers)):
        error = noise_layers[i] - data_layers[i]
        errs.append(np.sum(np.square(error)) / error.size)
    try:
        epsilon = 1e-10
        eta = np.log(epsilon + np.sum(errs))
        if eta > 0:
            na = np.log(((channel / eta) * (n_conv / len(errs))))
        else:
            na = 0

    except:
        na = 0

    del(noise_layers)
    del(data_layers)
    del(n1)
    del(n2)

    torch.cuda.empty_cache()
    return na

@torch.no_grad()
def old_ni_score(network, train_loader, device, args):
    network = network.to(device)
    data_iter = iter(train_loader)
    x, target = next(data_iter)
    x, target = x.to(device), target.to(device)

    noise = x.new(x.size()).normal_(0, args.sigma).to(device)
    x2 = x + noise

    o, _ = network(x)
    o_, _ = network(x2)
    o = o.detach().cpu().numpy()
    o_ = o_.detach().cpu().numpy()
    del network
    torch.cuda.empty_cache()

    return -np.sum(np.square(o-o_))

