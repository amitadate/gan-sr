import torch
import torch.nn as nn

'''
The discriminator network:
    1. Use LeakyReLU activation
    2. The size of input should be divided by 4
'''


class Adversarial_D2(nn.Module):
    def __init__(self, input_size):
        super(Adversarial_D2, self).__init__()
        
        kernel_size = int(input_size / 4) #otherwise it would give float  
        
        self.main = nn.Sequential(
            nn.Conv2d(1,64,3,stride = 2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(64,64,3,stride = 2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(64,1,kernel_size),
            nn.Sigmoid(),
        )


    def forward(self, inputs):
        outputs = self.main(inputs)
        return outputs.view(-1)
    