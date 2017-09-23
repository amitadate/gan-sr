import torch
import torch.nn as nn
import torch.nn.functional as F

'''
isize : image size，  
nz : number of latent variables
nc : number of channel
ngf : number of hidden units of generator

the size of input image : nz 
the size of output image : nc * isize * isize 

In this experiment : 
ngf = 64
nz = 14 * 14
nc = 1
isize = 28
'''
class MLP_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf):
        super(MLP_G, self).__init__()

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, nc * isize * isize),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0), -1)
        output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)