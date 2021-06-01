import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50,
             clamp_min=0.0, clamp_max=1.0, distance="l_2"):

    """
    Standard Deepfool (L_2 norm based)

       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :param clamp_min: the minimum of the value of the image
       :param clamp_max: the maximum of the value of the image
       :param distance: l_2 or l_inf
       :return: perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    if distance == 'l_2':
        print("l_2 based deepfool is running")
    elif distance == 'l_inf':
        print("l_inf based deepfool is running")
    else:
        assert False

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
            if distance == 'l_2':
                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
            elif distance == 'l_inf':
                pert_k = abs(f_k) / np.sum(np.abs(w_k.flatten()))
            else:
                assert False
            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        if distance == 'l_2':
            r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        elif distance == 'l_inf':
            r_i =  (pert+1e-4) * np.sign(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        loop_i += 1
    pert_image = torch.clamp(pert_image, clamp_min, clamp_max)

    return pert_image