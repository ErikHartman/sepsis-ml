
import collections 
import torch.nn as nn
from ReactomeNetwork import ReactomeNetwork
import torch.nn.utils.prune as prune
import torch
import numpy as np
from numpy import random

layers= nn.Sequential(
  nn.Linear(in_features=843, out_features=1034, bias=False)
  ,nn.Tanh()
  , nn.Linear(in_features=1034, out_features=473, bias=False)
  , nn.Tanh(),
   nn.Linear(in_features=473, out_features=162, bias=False)
 , nn.Tanh()
 ,nn.Linear(in_features=162, out_features=28, bias=False)
 , nn.Tanh(),
  nn.Linear(in_features=28, out_features=2, bias=True)
)

def residual_forward(x, layers):
    out_layer = nn.LazyLinear(2)
    r = out_layer(x)

    print('r1 ', r)
    for l in layers:
        if isinstance(l, nn.Linear):
            x = l(x) # linear 
        if isinstance(l,nn.Tanh) or isinstance(l, nn.ReLU):
            x = l(x) # activation
        out_layer = nn.LazyLinear(2)
        print('r2', out_layer(x))
        r += out_layer(x)
        print('r', r)
    # In vision this is solved by convolution, but I don't think it makes sense in our case (since neighbors aren't related).
    return r

input = random.rand(843)
input = torch.tensor(input).float()
r = residual_forward(input, layers)
print(r)