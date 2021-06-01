import torch
import torch.nn as nn

class StereoJoin(nn.Module):
    def forward(self, input):
        input_L = input[::2]
        input_R = input[1::2]
        tmp = input_L * input_R
        return torch.sum(tmp, 1,keepdim = True)