import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os
#import pandas as pd


def LFR_loss(model,
             x_natural,
             y,
             lambada,
             optimizer,
             criterion=nn.CrossEntropyLoss(),
             step_size=0.007,
             epsilon=0.031,
             perturb_steps=7):
 
    model.eval()
    
    # Generate adversarial samples
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_adv = torch.clamp(x_adv,0.0,1.0)

    i = 0
    for _ in range(perturb_steps):
        i = i+1
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_adv = criterion(model(x_adv), y)     
        x_adv_gradient = torch.autograd.grad(loss_adv, x_adv, create_graph=True, retain_graph=True)
        with torch.enable_grad():
            #### calculate "gradient loss"
            loss_grad = torch.mean(torch.sum(torch.abs(x_adv_gradient[0]),dim=(1,2,3)))
        #### update x_adv using the grad of "gradient loss" (???? not sure for this operation)
        gradient_for_update = torch.autograd.grad(loss_grad, x_adv, create_graph=False)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(gradient_for_update.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    
    ##########
    optimizer.zero_grad()
    #print("model(x_natural)",model(x_natural).shape)
    loss_normal = criterion(model(x_natural), y)
    print("loss normal", loss_normal)
    
    #### claculate gradient loss(LFR) under the train mode
    x_adv.requires_grad_()
    loss_adv = criterion(model(x_adv), y)
    x_adv_gradient = torch.autograd.grad(loss_adv, x_adv, create_graph=True, retain_graph=True)[0]
    loss_grad = torch.mean(torch.sum(torch.abs(x_adv_gradient),dim=(1,2,3)))
    ### multiply batch size

    loss_grad = loss_grad * x_adv.size(0)
    print("loss grad in iteration", loss_grad)
    
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    grad_part = lambada * loss_grad
    loss_adversarial = loss_normal + grad_part
    print("loss_adversarial",loss_adversarial)
  
    return loss_adversarial, loss_normal, loss_grad,grad_part



