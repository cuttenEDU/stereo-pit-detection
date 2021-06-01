import torch
import torch.nn as nn

class Normalize(nn.Module):

    def forward(self, input):
        norm = torch.sum(torch.square(input),1).unsqueeze(1) + 1e-5
        return input/torch.sqrt(norm)