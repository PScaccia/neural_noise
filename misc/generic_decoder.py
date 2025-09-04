#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:44:43 2025

@author: paolos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerExpNormNet(nn.Module):
    def __init__(self, N, M):
        super(TwoLayerExpNormNet, self).__init__()
        
        # First linear layer: input -> hidden
        self.fc1 = nn.Linear(N, M, bias=True)
        
        # Second linear layer: hidden -> output
        self.fc2 = nn.Linear(M, 1, bias=True)
        
    def forward(self, x):
        z = self.fc1(x)                 # linear transform
        hidden = F.softmax(z, dim=1)    # exp + normalization
        output = self.fc2(hidden)       # linear output
        return output

# Example usage:
N = 3   # input neurons
M = 4   # hidden neurons
model = TwoLayerExpNormNet(N, M)

# ---- Manually set weights and biases ----
# Hidden layer weights (M x N)
W1 = torch.tensor([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6],
                   [0.7, 0.8, 0.9],
                   [1.0, 1.1, 1.2]], dtype=torch.float32)

b1 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

# Assign to model
with torch.no_grad():
    model.fc1.weight.copy_(W1)   # shape [M, N]
    model.fc1.bias.copy_(b1)     # shape [M]
    
# Output layer weights (1 x M)
W2 = torch.tensor([[1.0, -1.0, 0.5, 0.2]], dtype=torch.float32)
b2 = torch.tensor([0.0], dtype=torch.float32)

with torch.no_grad():
    model.fc2.weight.copy_(W2)   # shape [1, M]
    model.fc2.bias.copy_(b2)     # shape [1]

# ---- Test the model ----
x = torch.tensor([[1.0, 2.0, 3.0]])  # single input sample
y = model(x)

print("Hidden responses (softmax normalized):", F.softmax(model.fc1(x), dim=1))
print("Final output:", y)