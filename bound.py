import torch
import torch.nn as nn
import numpy as np
import math

def max_pixel_sums(training_loader):
    # Initialize a tensor to hold the pixel sums
    pixel_sums = None

    # Loop over all batches in the training loader
    for batch_idx, (data, target) in enumerate(training_loader):
        # Convert the batch of images to a tensor
        if pixel_sums is None:
            pixel_sums = torch.zeros_like(data[0])

        # Sum the squares of the pixels
        for image in data:
            pixel_sums += torch.pow(image, 2)

    # Return the maximum pixel sum
    return pixel_sums.max().item()

def eval_rho1(net):
    rho = 1
    v = net['conv1.weight_v']
    g = net['conv1.weight_g']
    conv1_weight = g * (v / v.norm())
    rho *= conv1_weight.norm().item()
    v = net['conv2.weight_v']
    g = net['conv2.weight_g']
    conv2_weight = g * (v / v.norm())
    rho *= conv2_weight.norm().item()
    rho *= net['fc1.weight'].norm().max().item()
    rho *= net['fc2.weight'].norm().max().item()
    rho *= net['fc3.weight'].norm().max().item()
    return rho

def eval_rho2(net):
    rho = 1
    v = net['conv1.weight_v']
    g = net['conv1.weight_g']
    conv1_weight = g * (v / v.norm())
    rho *= conv1_weight.norm().item()
    v = net['conv2.weight_v']
    g = net['conv2.weight_g']
    conv2_weight = g * (v / v.norm())
    rho *= conv2_weight.norm().item()
    rho *= net['fc1.weight'].norm(dim=1).max().item()
    rho *= net['fc2.weight'].norm(dim=1).max().item()
    rho *= net['fc3.weight'].norm(dim=1).max().item()
    return rho

def eval_rho3(net):
    rho = 1
    v = net['conv1.weight_v']
    g = net['conv1.weight_g']
    conv1_weight = g * (v / v.norm())
    rho *= conv1_weight.norm(dim=(2,3)).max().item()
    v = net['conv2.weight_v']
    g = net['conv2.weight_g']
    conv2_weight = g * (v / v.norm())
    rho *= conv2_weight.norm(dim=(2,3)).max().item()
    rho *= net['fc1.weight'].norm(dim=1).max().item()
    rho *= net['fc2.weight'].norm(dim=1).max().item()
    rho *= net['fc3.weight'].norm(dim=1).max().item()
    return rho

def bound(rho, net, training_loader, max_pixel_sums):
    n = len(training_loader.dataset)
    k = 10
    depth = 7
    degs = [5**2,3**2,5**2,3**2, 192*6*6, 384, 192]
    delta = 0.001
    deg_prod = np.prod(degs)
    mult1 = (2 ** 1.5) * (rho + 1) / n
    mult2 = 1 + math.sqrt(2 * (np.log(2) * depth + np.log(deg_prod) + np.log(k)))
    mult3 = math.sqrt(max_pixel_sums * deg_prod)
    add1 = 3 * math.sqrt(np.log((2 * (rho + 2) ** 2) / delta) / (2 * n))
    bound = mult1 * mult2 * mult3 + add1
    return bound

def calc_bounds(net, training_loader, max_pixel_sums):
    rho1 = eval_rho1(net)
    rho2 = eval_rho2(net)
    rho3 = eval_rho3(net)
    bound1 = bound(rho1, net, training_loader, max_pixel_sums)
    bound2 = bound(rho2, net, training_loader, max_pixel_sums)
    bound3 = bound(rho3, net, training_loader, max_pixel_sums)
    return bound1, bound2, bound3