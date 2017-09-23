import torch
import torch.nn as nn
'''
The input_size should be divided by 32
'''

class Adversarial_D(nn.Module):
    def __init__(self, input_size):
        super(Adversarial_D, self).__init__()

        kernel_size = int(input_size / 32)

        self.main = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding =1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(64,64,3,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64,128,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128,128,3,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128,256,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256,256,3,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256,512,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,512,3,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,1024,3,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
       
            nn.Conv2d(1024,1,kernel_size),
            nn.Sigmoid(),
        )

    def forward(self, x):

        out = self.main(x)
        return out.view(-1)



