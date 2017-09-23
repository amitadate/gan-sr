import torch
import torch.nn as nn

'''
The discriminator network:
    1. Use LeakyReLU activation
    2. The size of input should be divided by 4
'''


class Adversarial_D(nn.Module):
    def __init__(self, input_size, wgan = False):
        super(Adversarial_D, self).__init__()
        self.wgan = wgan
        kernel_size = int(input_size / 4) #otherwise it would give float  
        
        self.main = nn.Sequential(
            nn.Conv2d(3,64,3,stride = 2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(64,64,3,stride = 2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(64,1,kernel_size),
        )

    def forward(self, inputs):
        outputs = self.main(inputs)
        outputs = outputs.view(-1)
        if self.wgan == False:
            s = nn.Sigmoid()
            outputs = s(outputs)
        return outputs
    