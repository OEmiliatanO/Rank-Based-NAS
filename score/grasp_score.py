# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy

from utils import get_layer_metric_array

def grasp_score(net, train_loader, device, args, mode = 'param', loss_fn = nn.CrossEntropyLoss()):
    net = copy.deepcopy(net)
    net = net.to(device)

    # get all applicable weights
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True) # TODO isn't this already true?

    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)
    net.zero_grad()
    #forward/grad pass #1
    grad_w = None
    outputs = net.forward(data)
    if isinstance(outputs, tuple):
        if args.nasspace == 'nasbench201' or args.nasspace == 'nasbench101':
            outputs = outputs[0]
        elif args.nasspace == 'natsbenchsss':
            outputs = outputs[1]
    loss = loss_fn(outputs, targets)
    grad_w_p = autograd.grad(loss, weights, allow_unused=True)
    if grad_w is None:
        grad_w = list(grad_w_p)
    else:
        for idx in range(len(grad_w)):
            grad_w[idx] += grad_w_p[idx]

    # forward/grad pass #2
    outputs = net.forward(data)
    if isinstance(outputs, tuple):
        if args.nasspace == 'nasbench201' or args.nasspace == 'nasbench101':
            outputs = outputs[0]
        elif args.nasspace == 'natsbenchsss':
            outputs = outputs[1]
    loss = loss_fn(outputs, targets)
    grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
        
    # accumulate gradients computed in previous step and call backwards
    z, count = 0,0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if grad_w[count] is not None:
                z += (grad_w[count].data * grad_f[count]).sum()
            count += 1
    z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad   # -theta_q Hg
            #NOTE in the grasp code they take the *bottom* (1-p)% of values
            #but we take the *top* (1-p)%, therefore we remove the -ve sign
            #EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)
    
    grads = get_layer_metric_array(net, grasp, mode)

    ret = 0
    for v in grads:
        v = v.detach()
        ret += torch.sum(v)

    ret = ret.detach().cpu().numpy()
    loss.detach()

    for v0, v1, v2 in zip(grad_f, grad_w, grad_w_p):
        if v0 != None: v0.detach()
        if v1 != None: v1.detach()
        if v2 != None: v2.detach()

    z.detach()
    del net, weights, grads, data, targets, grad_f, grad_w, grad_w_p, outputs, z

    return ret
