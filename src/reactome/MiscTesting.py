
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
    out_act = nn.Tanh()
    r = out_layer(x)
    r = out_act(r) # output from first layer
    for l in layers:
        if isinstance(l, nn.Linear):
            x = l(x) # linear 
        if isinstance(l, nn.BatchNorm1d):
            x = l(x)
        if isinstance(l,nn.Tanh) or isinstance(l, nn.ReLU):
            x = l(x) # activation
        out_layer = nn.LazyLinear(2)
        r2 = out_layer(x)
        out_act = nn.Tanh()
        r += out_act(r2)
    return r


input = random.rand(843)
input = torch.tensor(input).float()
out = layers(input)
print(out)