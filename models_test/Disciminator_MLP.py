import torch
import torch.nn as nn
'''
The discriminator network outputs a probability for each input

In this experiment : 
isize : 28
nc : 1
ndf : 64
'''

class MLP_D(nn.Module):
    def __init__(self, isize, nc, ndf):
        super(MLP_D, self).__init__()

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
            nn.Sigmoid() # make the outputs to be in [0,1], so that it represents a probability
        )
        self.main = main
        self.nc = nc
        self.isize = isize

    def forward(self, input):
        input = input.view(input.size(0),-1)
        output = self.main(input)

        return output