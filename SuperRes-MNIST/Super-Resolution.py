# for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms

# for os and plot 
import os
import shutil
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# to record time
import time

# for my models
from models import *

# for dataset
import mycifar

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'cifar', help='cifar10')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--Diters', type=int, default=1, help='number of iteration of discriminator per iteration of generator')
parser.add_argument('--lrG', type=float, default=0.01, help='learning rate of generator, default=0.01')
parser.add_argument('--lrD', type=float, default=0.01, help='learning rate of discriminator, default=0.01')
parser.add_argument('--ratio', type=float, default=1e-2, help = 'ratio of GAN loss')
parser.add_argument('--mode', type = str, default = 'MSE', help = 'MSE | GAN | WGAN | visual')
parser.add_argument('--resume', type = bool, default = True, help = 'True | False')

parser.add_argument('--G', default='G_best.pth.tar', help="path to netG (to continue training)")
parser.add_argument('--D', default='', help="path to netD (to continue training)")

opt = parser.parse_args()
print(opt)

# show a 3-channel image
def myimshow(npimg):
    img = npimg
    img = (img - img.min()) / (img.max() - img.min())
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.axis('off')
# some helper function
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_checkpoint(state, is_best, filename='G.pth.tar'):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    print('==> Saving checkpoint')
    torch.save(state, 'checkpoint/' + filename)
    if is_best:
        print('This is best checkpoint ever, copying')
        shutil.copyfile('checkpoint/'+filename, 'checkpoint/'+'G_best.pth.tar')
        
        
def train(train_loader, model, criterion, optimizer, epoch):
    print('==> Starting Training Epoch {}'.format(epoch))
    
    losses = AverageMeter()
    
    model.train()  # Set the model to be in training mode
    
    for batch_index, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # update loss
        losses.update(loss.data[0], inputs.size(0))
        
        # Backward
        optimizer.zero_grad()  # Set parameter gradients to zero
        loss.backward()        # Compute (or accumulate, actually) parameter gradients
        optimizer.step()       # Update the parameters
        
    print('==> Train Epoch : {}, Average loss is {}'.format(epoch, losses.avg))
    
def validate(validate_loader, model, criterion, epoch):   
    
    print('==> Starting validate')
    model.eval()
    
    losses = AverageMeter()
    
    correct = 0
    
    for batch_idx, (inputs, targets) in enumerate(validate_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # update loss, accuracy
        losses.update(loss.data[0], inputs.size(0))
        
    print('==> Validate Epoch : {}, Average loss, {:.4f}'.format(
        epoch,losses.avg))
    
    return losses.avg


def imshow(img):
    img = (img - img.min()) / (img.max() - img.min())
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')
    plt.show()
    


    
# GPU setting
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('==> GPU is available and will train with GPU')
else :
    print('==> GPU is not available and will train with CPU')

# set dataset
if opt.dataset == 'cifar':
    cifar = mycifar.CIFAR10('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    loader = torch.utils.data.DataLoader(cifar,batch_size=opt.batch_size, shuffle=True)
    validate_cifar = mycifar.CIFAR10('data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))
    validate_loader = torch.utils.data.DataLoader(validate_cifar,batch_size=opt.batch_size, shuffle=True)
    print('==> CIFAR have been set')
    
     
    
# train with MSE loss

if opt.mode == 'MSE':
    # parameters
    start_epoch = 0
    best_prec1 = 10   
    betas = (0.9, 0.999)
    
    # network
    G = Adversarial_G()
    optimizerG = torch.optim.Adam(G.parameters(), lr = opt.lrG, betas = betas, weight_decay=5e-4)
    criterionG = nn.MSELoss()
    
    # resume
    if opt.resume:
        if os.path.isfile('checkpoint/'+ 'G_best.pth.tar'):
            print("==> loading checkpoint {}".format('G_best.pth.tar'))
            checkpoint = torch.load('checkpoint/'+ 'G_best.pth.tar')
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            G.load_state_dict(checkpoint['state_dict'])
            print('==> {} has been loaded'.format('G_best.pth.tar'))
        else:
            print("=> no checkpoint found")
        print('\n')
    
    if use_cuda:
        G.cuda()
        criterionG.cuda()
                 
    
    print('==> Train with MSE loss')
    print('==========================')
    
    train_time = AverageMeter()
    validate_time = AverageMeter()

    for epoch in range(start_epoch, opt.niter):
        # train for one epoch
        end = time.time()
        train(loader, G, criterionG, optimizerG, epoch)
        train_time.update(time.time() - end)
        end = time.time()
        prec1 = validate(validate_loader, G, criterionG, epoch)
        validate_time.update(time.time() -end)
            

        # remember minimal loss and save checkpoint
        # only G network
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': G.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
                  
    print('Training with MSE is done')
    print('Average training time : {}'.format(train_time.avg))
    print('Average validate time : {}'.format(validate_time.avg))              
    print('Save the ouputs : ')
    G.eval()
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index == 0:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            G_outputs = G(inputs)
            
            torchvision.utils.save_image(inputs.data, 'inputs_samples_{}_{}_{}_{}_{}.png'.format(opt.mode, opt.niter, opt.lrD, opt.lrG, opt.ratio))
            torchvision.utils.save_image(G_outputs.data, 'outputs_samples_{}_{}_{}_{}_{}.png'.format(opt.mode, opt.niter, opt.lrD, opt.lrG, opt.ratio))
            torchvision.utils.save_image(targets.data, 'origin_samples_{}_{}_{}_{}_{}.png'.format(opt.mode, opt.niter,opt.lrD, opt.lrG, opt.ratio))

if opt.mode == 'WGAN':
    print('I love GAN\n')
    
    input_size = 32
    # networks
    G = Adversarial_G()
    D = Adversarial_D(input_size = input_size, wgan = True)
  
    optimizerG = torch.optim.RMSprop(G.parameters(), lr = opt.lrG)
    optimizerD = torch.optim.RMSprop(D.parameters(), lr = opt.lrD)
    
    criterionG = nn.MSELoss()
    
    if opt.resume:
        if os.path.isfile('checkpoint/'+ 'G_best.pth.tar'):
            print("==> loading checkpoint {}".format('G_best.pth.tar'))
            checkpoint = torch.load('checkpoint/'+ 'G_best.pth.tar')
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            G.load_state_dict(checkpoint['state_dict'])
            print('==> {} has been loaded'.format('G_best.pth.tar'))
        else:
            print("=> no checkpoint found")
        print('\n')
    
    one = torch.FloatTensor([1]*opt.batch_size)
    mone = one * -1

    if use_cuda:
        G.cuda()
        D.cuda()
        one, mone = one.cuda(), mone.cuda()
          
    isstart = True # at first, train D with more iterations
    for epoch in range(opt.niter):
        print('==> Start training epoch {}'.format(epoch))
        
        data_iter = iter(loader)
        i = 0
        while i < len(loader):
            # update D network
            for p in D.parameters():
                p.requires_grad = True
            if isstart:
                Diters = 500
                isstart = False
            else :
                Diters = opt.Diters
                
            j=0
            while j < Diters and i < len(loader) - 1:
                j+=1
                
                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01)
                    
                data = data_iter.next()
                i+=1
                
                # train with real
                D.zero_grad()        
                small_img, origin_img = data
                if use_cuda:
                    origin_img = origin_img.cuda()
                    small_img = small_img.cuda()
                
                # train with real
                inputv = Variable(origin_img)
                errD_real = D(inputv)
                errD_real.backward(one)
                
                # train with fake
                noisev = Variable(small_img, volatile = True) # totally freeze G
                fake = Variable(G(noisev).data)
                inputv = fake
                errD_fake = D(inputv)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()
            
            # update G
            for p in D.parameters():
                p.requires_grad = False # to avoid computation
            
            G.zero_grad()
            
            data = data_iter.next()
            i += 1 

                
            small_img, origin_img = data
            if small_img.size(0) != opt.batch_size:
                continue 

            if use_cuda:
                origin_img = origin_img.cuda()
                small_img = small_img.cuda()
                    
            small_img_var, origin_img_var = Variable(small_img), Variable(origin_img)
            fake = G(small_img_var)
            errG_GAN = D(fake)
            errG_GAN.backward(one, retain_variables=True)
                      
            #errG_content = criterionG(fake, origin_img_var)
            #errG_content.backward()
                
            #errG = errG_GAN + errG_content

            optimizerG.step()
            
    torch.save({'epoch': epoch + 1,
                'state_dict': G.state_dict(),
                'best_prec1': 0,
               }, 'checkpoint/G_WGAN.pth.tar' )
    torch.save({'epoch': epoch + 1,
                'state_dict': D.state_dict(),
                'best_prec1': 0,
               }, 'checkpoint/D_WGAN.pth.tar' )
    
    print('Save the ouputs : ')
    G.eval()
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index == 0:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            G_outputs = G(inputs)
            
            torchvision.utils.save_image(inputs.data, 'inputs_samples_{}_{}_{}_{}_{}.png'.format(opt.mode, opt.niter, opt.lrD, opt.lrG, opt.ratio))
            torchvision.utils.save_image(G_outputs.data, 'outputs_samples_{}_{}_{}_{}_{}.png'.format(opt.mode, opt.niter, opt.lrD, opt.lrG, opt.ratio))
            torchvision.utils.save_image(targets.data, 'origin_samples_{}_{}_{}_{}_{}.png'.format(opt.mode, opt.niter,opt.lrD, opt.lrG, opt.ratio))
       
elif opt.mode == 'GAN':
    print('I love GAN\n')
    
    input_size = 32
    # networks
    G = Adversarial_G()
    D = Adversarial_D(input_size = input_size, wgan=False)
  
    optimizerG = torch.optim.Adam(G.parameters(), lr = opt.lrG)
    optimizerD = torch.optim.Adam(D.parameters(), lr = opt.lrD)
    
    criterionG = nn.MSELoss()
    criterionD = nn.BCELoss()
    
    if opt.resume:
        if os.path.isfile('checkpoint/'+ 'G_best.pth.tar'):
            print("==> loading checkpoint {}".format('G_best.pth.tar'))
            checkpoint = torch.load('checkpoint/'+ 'G_best.pth.tar')
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            G.load_state_dict(checkpoint['state_dict'])
            print('==> {} has been loaded'.format('G_best.pth.tar'))
        else:
            print("=> no checkpoint found")
        print('\n')
    


    # labels 
    labels = torch.FloatTensor(opt.batch_size) 
            
    real_label = 1
    fake_label = 0

    if use_cuda:
        G.cuda()
        D.cuda()
        labels = labels.cuda()

    labels = Variable(labels)

    isstart = True    
    for epoch in range(opt.niter):
        print('==> Start training epoch {}'.format(epoch))
        
        data_iter = iter(loader)
        i = 0
        while i < len(loader):
            # update D network
            for p in D.parameters():
                p.requires_grad = True
            if isstart:
                Diters = 500
                isstart = False
            else :
                Diters = opt.Diters
                
            j=0
            while j < Diters and i < len(loader)-1:
                j+=1
                    
                data = data_iter.next()
                i+=1                
                # train with real
                D.zero_grad()        
                small_img, origin_img = data
                if use_cuda:
                    origin_img = origin_img.cuda()
                    small_img = small_img.cuda()
                
                inputv = Variable(origin_img)
                output_real = D(inputv)
                labels.data.resize_(output_real.size()).fill_(real_label)
                errD_real = criterionD(output_real, labels)
                errD_real.backward()
                
                # train with fake
                noisev = Variable(small_img, volatile = True) # totally freeze G
                fake = Variable(G(noisev).data)
                inputv = fake
                output_fake = D(inputv)
                labels.data.resize_(output_fake.size()).fill_(fake_label)
                errD_fake = criterionD(output_fake, labels)     
                errD_fake.backward()

                errD = errD_real + errD_fake
                optimizerD.step()

            
            # update G  
            for p in D.parameters():
                p.requires_grad = False # to avoid computation
                        
            G.zero_grad()
            data = data_iter.next()
            i += 1 
                
            small_img, origin_img = data
            if small_img.size(0) != opt.batch_size:
                continue
                
            if use_cuda:
                origin_img = origin_img.cuda()
                small_img = small_img.cuda()
            small_img_var, origin_img_var = Variable(small_img), Variable(origin_img)
            fake = G(small_img_var)   
            output = D(fake)
            labels.data.resize_(output.size()).fill_(real_label) # fake labels are real for generator cost

            errG_GAN = opt.ratio * criterionD(output, labels)
            errG_GAN.backward(retain_variables=True)
 
            errG_content = criterionG(fake, origin_img_var)
            errG_content.backward()
                
            errG = errG_GAN + errG_content

            optimizerG.step()


    torch.save({'epoch': epoch + 1,
                'state_dict': G.state_dict(),
                'best_prec1': 0,
               }, 'checkpoint/G_GAN.pth.tar' )
    torch.save({'epoch': epoch + 1,
                'state_dict': D.state_dict(),
                'best_prec1': 0,
               }, 'checkpoint/D_GAN.pth.tar' )


    print('Save the ouputs : ')
    G.eval()
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index == 0:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            G_outputs = G(inputs)
            
            torchvision.utils.save_image(inputs.data, 'inputs_samples_{}_{}_{}_{}_{}.png'.format(opt.mode, opt.niter, opt.lrD, opt.lrG, opt.ratio))
            torchvision.utils.save_image(G_outputs.data, 'outputs_samples_{}_{}_{}_{}_{}.png'.format(opt.mode, opt.niter, opt.lrD, opt.lrG, opt.ratio))
            torchvision.utils.save_image(targets.data, 'origin_samples_{}_{}_{}_{}_{}.png'.format(opt.mode, opt.niter,opt.lrD, opt.lrG, opt.ratio))
       

if opt.mode == 'visual':

    G_MSE = Adversarial_G()
    G_GAN = Adversarial_G()
    G_WGAN = Adversarial_G()

    if opt.resume:
        if os.path.isfile('checkpoint/'+ 'G_best.pth.tar'):
            print("==> loading checkpoint {}".format('G_best.pth.tar'))
            checkpoint = torch.load('checkpoint/'+ 'G_best.pth.tar')
            G_MSE.load_state_dict(checkpoint['state_dict'])
            print('==> {} has been loaded'.format('G_best.pth.tar'))
        else:
            print("=> no checkpoint for MSE found")

        if os.path.isfile('checkpoint/'+ 'G_GAN.pth.tar'):
            print("==> loading checkpoint {}".format('G_GAN.pth.tar'))
            checkpoint = torch.load('checkpoint/'+ 'G_GAN.pth.tar')
            G_GAN.load_state_dict(checkpoint['state_dict'])
            print('==> {} has been loaded'.format('G_GAN.pth.tar'))
        else:
            print("=> no checkpoint for GAN found")

        if os.path.isfile('checkpoint/'+ 'G_WGAN.pth.tar'):
            print("==> loading checkpoint {}".format('G_WGAN.pth.tar'))
            checkpoint = torch.load('checkpoint/'+ 'G_WGAN.pth.tar')
            G_WGAN.load_state_dict(checkpoint['state_dict'])
            print('==> {} has been loaded'.format('G_WGAN.pth.tar'))
        else:
            print("=> no checkpoint for WGAN found")    
        print('\n')
        
    G_MSE.eval()
    G_GAN.eval()
    G_WGAN.eval()
    if use_cuda:
        G_MSE.cuda()
        G_GAN.cuda()
        G_WGAN.cuda()


    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index == 0:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs_MSE = G_MSE(inputs)
            outputs_GAN = G_GAN(inputs)
            outputs_WGAN = G_WGAN(inputs)


            # LR / MSE / GAN / WGAN / original 
            # four images 
            torchvision.utils.save_image(targets.data, 'origin_samples_origin.png')
            torchvision.utils.save_image(outputs_MSE.data, 'origin_samples_MSE.png')
            torchvision.utils.save_image(outputs_GAN.data, 'origin_samples_GAN.png')
            torchvision.utils.save_image(outputs_WGAN.data, 'origin_samples_WGAN.png')









            
            


        
