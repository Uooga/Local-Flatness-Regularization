"""
Adversarial Defense via Local Flatness Regularization

@author: Jia Xu; Yiming Li
@mails: xujia19@mails.tsinghua.edu.cn;li-ym18@mails.tsinghua.edu.cn
"""
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import random
import foolbox

from advertorch.utils import predict_from_logits
import models.cifar as models
import torch.backends.cudnn as cudnn

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MNIST Robustness Evaluation')
parser.add_argument('--model_path', default='', help='trained model path')
parser.add_argument('--attack_method', type=str, default='PGD', help='adversarial attack method, including FGSM, PGD')
parser.add_argument('--test_batch', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--gpu_id', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Target Model Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block_name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen_factor', type=int, default=6, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# torch.manual_seed(666)
attack_mode = args.attack_method

# Target Model
print("==> creating model '{}'".format(args.arch))
if args.arch.endswith('resnet'):
    model = models.__dict__[args.arch](
        num_classes=10,
        depth=args.depth,
        block_name=args.block_name,
    )
elif args.arch.endswith('sig'):
    model = models.__dict__[args.arch](
        num_classes=10,
        depth=args.depth,
        block_name=args.block_name,
    )
elif args.arch.endswith('tanh'):
    model = models.__dict__[args.arch](
        num_classes=10,
        depth=args.depth,
        block_name=args.block_name,
    )
elif args.arch.endswith('gn'):
    model = models.__dict__[args.arch](
        num_classes=10,
        depth=args.depth,
        block_name=args.block_name,
    )
else:
    model = models.__dict__[args.arch](num_classes=10)

model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

print('==> Load the target model..')
assert os.path.isfile(args.model_path), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=args.test_batch, shuffle=False,
    pin_memory=True
)

print('==> This is the %s attack' % args.attack_method)

if attack_mode == 'PGD':
    from advertorch.attacks import LinfPGDAttack

    print("it is pgd attack")
    adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031,
                              nb_iter=10, eps_iter=0.007, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
elif attack_mode == 'MPGD':
    from advertorch.attacks import LinfMomentumIterativeAttack

    adversary = LinfMomentumIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031,
                                            decay_factor=0.1,
                                            nb_iter=20, eps_iter=0.03, clip_min=0.0, clip_max=1.0, targeted=False)
elif attack_mode == 'FGSM':
    from advertorch.attacks import LinfPGDAttack

    adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                              nb_iter=1, eps_iter=0.3, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
elif attack_mode == 'JSMA':
    from advertorch.attacks import JacobianSaliencyMapAttack

    adversary = JacobianSaliencyMapAttack(model, num_classes=10, clip_min=0., clip_max=1., gamma=0.4)
elif attack_mode == 'DDNL2':
    from advertorch.attacks import DDNL2Attack

    adversary = DDNL2Attack(model, nb_iter=20)
elif attack_mode == 'point-wise':
    from foolbox.attacks import PointwiseAttack

    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
    adversary = PointwiseAttack(fmodel)
else:
    print ('Not Implemented %s' % (attack_mode))
    assert (False)

correct_clean = 0
correct_adv = 0

for idx, (cln_data, true_label) in enumerate(loader):
    if args.attack_method == 'point-wise':
        adv_untargeted = adversary(cln_data.numpy(), true_label.numpy())
        adv_untargeted = torch.from_numpy(adv_untargeted).to(device)
        cln_data = cln_data.to(device)
        true_label = true_label.to(device)
    else:
        cln_data, true_label = cln_data.to(device), true_label.to(device)
        if args.attack_method == 'JSMA':
            adv_untargeted = adversary.perturb(cln_data,
                                               (torch.ones_like(true_label) * int(random.random() * 10)).to(device))
        else:
            adv_untargeted = adversary.perturb(cln_data, true_label)

    pred_cln = predict_from_logits(model(cln_data))
    pred_adv = predict_from_logits(model(adv_untargeted))

    correct_clean = correct_clean + (pred_cln.data == true_label.data).float().sum()
    correct_adv = correct_adv + (pred_adv.data == true_label.data).float().sum()

    print("current correct clean samples: %s; current correct adv samples: %s" % (
    correct_clean.data.item(), correct_adv.data.item()))

print("correct clean samples: ", correct_clean)
print("correct adversarial samples: ", correct_adv)



