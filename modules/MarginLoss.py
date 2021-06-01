import torch.nn as nn
from adcensus.cupy_adcensus import *

class MarginLoss(nn.Module):
    def __init__(self,margin):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, input):
        in_pos = input[::2]
        in_neg = input[1::2]
        zero = torch.Tensor([0]).to(input.device)
        return torch.mean(torch.max(zero,self.margin+in_neg-in_pos))
