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

import types
import copy

from utils import get_layer_metric_array, reshape_elements

def fisher_forward_conv2d(self, x):
    x = F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
    #intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act

def fisher_forward_linear(self, x):
    x = F.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act

def fisher_score(net, train_loader, device, args, loss_fn = nn.CrossEntropyLoss(), mode = 'channel'):
    net = copy.deepcopy(net)
    net = net.to(device)
    net.train()
    print(f"in fisher, after to device: {torch.cuda.memory_allocated()}")
    all_hooks = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            #variables/op needed for fisher computation
            layer.fisher = None
            layer.act = 0.
            layer.dummy = nn.Identity()

            #replace forward method of conv/linear
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(fisher_forward_linear, layer)

            #function to call during backward pass (hooked on identity op at output of layer)
            def hook_factory(layer):
                def hook(module, grad_input, grad_output):
                    act = layer.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        g_nk = torch.sum((act * grad), list(range(2,len(act.shape))))
                    else:
                        g_nk = act * grad
                    del_k = g_nk.pow(2).mean(0).mul(0.5)
                    if layer.fisher is None:
                        layer.fisher = del_k
                    else:
                        layer.fisher += del_k
                    del layer.act #without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                return hook
            #register backward hook on identity fcn to compute fisher info
            all_hooks.append(layer.dummy.register_backward_hook(hook_factory(layer)))

    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)
    net.zero_grad()
    print(f"in fisher, before net(data): {torch.cuda.memory_allocated()}")
    outputs = net(data)
    print(f"in fisher, after net(data): {torch.cuda.memory_allocated()}")
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = loss_fn(outputs, targets)
    loss.backward()

    print(f"in fisher, after loss backward: {torch.cuda.memory_allocated()}")
    # retrieve fisher info
    def fisher(layer):
        if layer.fisher is not None:
            layer.fisher = layer.fisher.detach()
            return torch.abs(layer.fisher)
        else:
            return torch.zeros(layer.weight.shape[0]) #size=ch
    
    print(f"1 in fisher: {torch.cuda.memory_allocated()}")
    grads_abs_ch = get_layer_metric_array(net, fisher, mode)

    #broadcast channel value here to all parameters in that channel
    #to be compatible with stuff downstream (which expects per-parameter metrics)
    #TODO cleanup on the selectors/apply_prune_mask side (?)
    shapes = get_layer_metric_array(net, lambda l : l.weight.shape[1:], mode)

    grads_abs = reshape_elements(grads_abs_ch, shapes, device)
    
    print(f"2 in fisher: {torch.cuda.memory_allocated()}")
    ret = 0
    for v0, v1 in zip(grads_abs, grads_abs_ch):
        v0 = v0.detach()
        v1 = v1.detach()
        ret += torch.sum(v0)

    ret = ret.detach().cpu().numpy()

    for hook in all_hooks:
        hook.remove()

    loss.detach()
    #data.detach()
    #targets.detach()
    #outputs.detach()
    #del net, grads_abs, shapes, grads_abs_ch, data, targets, outputs, loss, all_hooks
    del net, grads_abs_ch, shapes, grads_abs

    print(f"3 in fisher: {torch.cuda.memory_allocated()}")
    #print(f"{torch.cuda.memory_summary()}")
    #print(torch.cuda.memory_stats())
    return ret
