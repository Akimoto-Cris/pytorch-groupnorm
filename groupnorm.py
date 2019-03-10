import torch
import torch.nn as nn


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        
        if C % G == 0:
            x = x.view(N,G,-1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            x = (x-mean) / (var+self.eps).sqrt()
        else:
            whole = x[:, : G * (C // G), :, :]
            mod = x[:, G * (C // G) + 1:, :, :]
            whole_mean = whole.mean(-1, keepdim=True)
            whole_var = whole.var(-1, keepdim=True)
            mod_mean = mod.mean(-1, keepdim=True)
            mod_var = mod.var(-1, keepdim=True)
            whole, mod = tuple(map(lambda x, mean, var: (x-mean) / (var+self.eps).sqrt(), 
                                   [whole, mod], [whole_mean, mod_mean], [whole_var, mod_var]))
            x = torch.cat((whole, mod), axis=1)
        
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias
