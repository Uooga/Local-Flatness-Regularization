#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Defense via Local Flatness Regularization

@author: Jia Xu; Yiming Li
@mails: xujia19@mails.tsinghua.edu.cn;li-ym18@mails.tsinghua.edu.cn
"""
#test

from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from tensorboardX import SummaryWriter
from loss_function.LFR_loss import LFR_loss
from torch.nn.utils import clip_grad_norm


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR SRTD')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--optimizer_type', default='SGD', type=str,
                     help='optimizer type')
parser.add_argument('--change_opti', action='store_true', default=False, help='if True, then new opti')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--change_lr', action='store_true', default=False, help='if True, then use the new lr')

parser.add_argument('--drop', '--dropout', default=0., type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[75,90,100],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default=None, type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--save_dir', default='./1212/v6', type=str, metavar='PATH',
                    help='path to save checkpoint & other output material(default: /data/xujia/Stand')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt_interval', default='5', type=int,
                    help='save interval')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='wrn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--block_name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen_factor', type=int, default=6, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int,default=1000, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#gradient clip
parser.add_argument('--clip_grad', default=10, type=float,
                    help='maximum gradient for gradient clip')
parser.add_argument('--clip_norm', default=2, type=float,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#Define the number of labeled data
parser.add_argument('--nlabel', default=50000, type=int, help='number of labels')


#Hyperparameters in adversarial training
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation')#8/255
parser.add_argument('--num_steps', type=int, default=10, help='perturb number of steps')
parser.add_argument('--step_size', type=float, default=0.007, help='perturb step size')
parser.add_argument('--lambada', type=float, default=1.0, help='hyperparameter')
parser.add_argument('--norm', type=str, default='l_inf', help='Distance measure of the perturbation')

####schedule adjusting
parser.add_argument('--debug_mode', action='store_true', default=False, help='if True, then do not adjusting lr')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Validate norm
assert args.norm == 'l_2' or args.norm == 'l_inf', 'Distance measure can only be l_2 or l_inf.'


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy




def main():
    
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    
    # Show the method
    print('==> This is the LFR defense with %s norm' %( args.norm))
    
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),

    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    #Define labeled loader and test loader
    labeledset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    num_train = len(labeledset.data)
    labeledset.data = labeledset.data[0:args.nlabel]
    labeledset.targets = labeledset.targets[0:args.nlabel]
    print("num_train",num_train)
    prop_labeled = args.nlabel/num_train
    print("args.nlabel",args.nlabel)
    print("prop_labeled",prop_labeled)
    labeled_loader = data.DataLoader(labeledset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    print("num_test",len(testset))


    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnet'):
        print("this model is end with resnet")
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.endswith('wrn'):
        print("this model is end with wrn")
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor = args.widen_factor,
                    dropRate=args.drop)
    
    else:
        model = models.__dict__[args.arch](num_classes=num_classes, dropRate=args.drop)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    if args.optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.change_opti == False:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        writer =  SummaryWriter(logdir = os.path.join(args.checkpoint,"scalars"),comment="debug")
        if args.change_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    else:
        mkedirs(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        tb_dir = os.path.join(args.checkpoint,"scalars")
        writer = SummaryWriter(log_dir=tb_dir, comment="debug")
        
    
    print("args.checkpoint",args.checkpoint)


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
   
    # Train and val
    logger.write_txt(str(state))
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.debug_mode, logger)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc,loss_normal,loss_grad = train(args, model, labeled_loader, criterion, optimizer, epoch, use_cuda,writer)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        writer.add_scalars("Loss_epoch",{"train_loss":train_loss,"test_loss":test_loss,"loss_normal":loss_normal,"loss_grad":loss_grad},epoch)
        writer.add_scalars("Acc_epoch", {"train_Acc": train_acc, "test_acc": test_acc}, epoch)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        
        ###save model per ckpt_step epochs
        if (epoch+1)%args.ckpt_interval==0:
            state1 = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }
            filename = "epoch{ep}.pth.tar".format(ep = epoch+1)
            filepath = os.path.join(args.checkpoint, filename)
            torch.save(state1, filepath)

            
        

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.write_txt(str(state))
    logger.write_txt("\nBest acc: ")
    logger.write_txt(str(best_acc))
    logger.close()

    print('Best acc:')
    print(best_acc)





def train(args, model, labeled_loader, criterion, optimizer, epoch, use_cuda,writer):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_normal = AverageMeter()
    losses_grad = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(labeled_loader))
    
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        # measure data loading time
        data_time.update(time.time() - end)
             
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute loss

        loss, loss_normal, loss_grad,grad_part=LFR_loss(model=model,
                                                  x_natural=inputs,
                                                  y=targets,
                                                  criterion=criterion,
                                                  optimizer=optimizer,                    
                                                  step_size=args.step_size,
                                                  epsilon=args.epsilon,
                                                  perturb_steps=args.num_steps,
                                                  lambada = args.lambada)
        writer.add_scalars("Loss_iter",
                               { "Loss": loss,"Loss normal": loss_normal, "Loss gradient": loss_grad,"grad_part":grad_part}, batch_idx)

        # measure train accuracy and record loss
        outputs = model(inputs)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses_normal.update(loss_normal.item(),inputs.size(0))
        losses_grad.update(loss_grad.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #clip_grad_norm(model.parameters(), max_norm=args.clip_grad, norm_type=args.clip_norm)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = 'epoch: ({ep}/{total_ep})|({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    ep=epoch + 1,
                    total_ep=args.epochs,
                    batch=batch_idx + 1,
                    size=len(labeled_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg,losses_normal.avg,losses_grad.avg)




def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()


    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output and standard loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record standard loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = 'epoch: ({ep}/{total_ep})|({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    ep=epoch + 1,
                    total_ep=args.epochs,
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)




def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def mkedirs(args):
    root_dir = args.save_dir
    print(root_dir)
    if not os.path.exists(root_dir):
        mkdir_p(root_dir)
    hyper_parameter_str = '{}/{}_{}_{}/lr_{}_lambada_{}_rs_{}'.format(args.dataset,args.arch,args.depth,args.widen_factor,
        args.lr, args.lambada,args.manualSeed)
    ckpt_path = os.path.join(root_dir,hyper_parameter_str)
    print(ckpt_path)
    if not os.path.exists(ckpt_path):os.makedirs(ckpt_path)
    args.checkpoint = ckpt_path


def adjust_learning_rate(optimizer, epoch, debug_mode,logger):
    global state
    if debug_mode == True:
        state['lr'] = state['lr']
        print("it is debug mode")
    else:
        if epoch in args.schedule:
            state['lr'] *= args.gamma
            logger.write_txt("state['lr']: ")
            logger.write_txt(str(state['lr']))
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']



if __name__ == '__main__':
    main()
