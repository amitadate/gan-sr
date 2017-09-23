import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The generator network : 
    1. Use pixel_shuffle to scale the image by 2X 
    2. Since all layers are convolutional layers, 
    so there is no requirement on the height and width to the images and
    the only requirement is the channel of the images is 1
'''

class Adversarial_G(nn.Module):
    def __init__(self):
        super(Adversarial_G, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32,4,3,padding=1),      
        )
        
    def forward(self, inputs):
        outputs = self.main(inputs)
        return F.pixel_shuffle(outputs, upscale_factor=2)
