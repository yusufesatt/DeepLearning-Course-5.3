# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:39:27 2023

@author: yusuf
"""

# %%
from torch.autograd import Variable
import torch
# %%
# define variable
var = Variable(torch.ones(3), requires_grad = True)
var

# %%

"""
Assume we have equation y = x^2
Define x = [2,4] variable
After calculation we find that y = [4,16] (y = x^2)
Recap o equation is that o = (1/2)sum(y) = (1/2)sum(x^2)
deriavative of o = x
Result is equal to x so gradients are [2,4]
Lets implement
"""

# lets make basic backward propagation
# we have an equation that is y = x^2
array = [2,4]
tensor = torch.Tensor(array)
x = Variable(tensor, requires_grad = True)
y = x**2
print(" y =  ",y)

# recap o equation o = 1/2*sum(y)
o = (1/2)*sum(y)
print(" o =  ",o)

# backward
o.backward() # calculates gradients

# As I defined, variables accumulates gradients. In this part there is only one variable x.
# Therefore variable x should be have gradients
# Lets look at gradients with x.grad
print("gradients: ",x.grad)