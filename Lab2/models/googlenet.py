import torch 
from torch import nn
import torch.nn.functional as F

class InceptionBlock(nn.Module): 
    def __init__(self, in_channels, ch1_):
        super().__init__()

        self.branch1 = nn.Sequential (
            nn.Conv2d (
                in_channels = in_channels,
                kernel_size = 1
            ),
            nn.ReLu()
        )

        self.branch2 = nn.Sequential (
            nn.Conv2d (
                in_channels = in_channels,
                out_channels = 3,
                padding = (1,1)
            )
        )

        self.branch3 = nn.Sequential (
            nn.Conv2d (
                in_channels = in_channels,
                
            )
        )