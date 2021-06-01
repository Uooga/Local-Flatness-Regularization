import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
from PIL import Image
from deepfool import deepfool

net = models.resnet34(pretrained=True)

# Switch to evaluation mode
net.eval()

im_orig = Image.open('test_im.jpg')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]


# Remove the mean
im = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

pert_image = deepfool(im, net, max_iter=30, distance="l_inf")
print("New Prediction :", np.argmax(net(pert_image).data.cpu().numpy().flatten()))
print("Previous Prediction:", np.argmax(net(im[None, :, :, :].cuda()).data.cpu().numpy().flatten()))

