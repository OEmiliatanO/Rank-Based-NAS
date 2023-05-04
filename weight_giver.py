import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

class Weight_giver(nn.Module):
    def __init__(self, N, M, Wn, learning_rate):
        super().__init__()
        self.flatten = nn.Flatten()
        self.module = nn.Sequential(
            nn.Linear(N*M, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, N*Wn)
        )
        self.N = N
        self.M = M
        self.Wn = Wn
        self.optimizer = torch.optim.Adadelta(self.parameters(), lr = learning_rate)

    def forward(self, x):
        #x = self.flatten(x)
        return self.module(x)
    
    def train(self, arch_str, indices, rank_fn, loss_fn, optimizer, device):
        x = torch.Tensor()
        for s in arch_str:
            x = torch.cat((x, torch.Tensor([c for c in s])))
        x = x.to(device)
        w = self.forward(x)
        w = w.detach().cpu().numpy()
        w.resize(self.N, self.Wn)
        a, b, c, d, e, f, g, h, maxacc, rk_maxacc = rank_fn(indices, w, len(arch_str))
        maxacc, rk_maxacc = Variable(torch.Tensor([maxacc for i in range(self.N * self.Wn)])), Variable(torch.Tensor([rk_maxacc for i in range(self.N * self.Wn)]))
        loss = loss_fn(maxacc, rk_maxacc)
        
        print("loss: ", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
