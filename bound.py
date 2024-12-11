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


def rankedness_by_var(X, threshold):

    # Suppose X is a PyTorch tensor of shape (n_samples, n_features)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    # S: [sigma_1, ... sigma_n] -> [sigma_1/trace(Sigma), ... , simga_n/trace(SIgma)]
    tot = S.sum()
    scaleyness = S/tot
    S_new = torch.cumsum(scaleyness, dim = 0)

    # Determine the number of components that explain at least 95% of the variance
    num_components = (S_new >= threshold).nonzero()[0].item() + 1
    return num_components


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
    # print(conv1_weight.shape)
    rho *= conv1_weight.norm(dim=(2,3)).max().item()
    v = net['conv2.weight_v']
    g = net['conv2.weight_g']
    conv2_weight = g * (v / v.norm())
    # print(conv2_weight.shape)
    rho *= conv2_weight.norm(dim=(2,3)).max().item()
    rho *= net['fc1.weight'].norm(dim=1).max().item()
    rho *= net['fc2.weight'].norm(dim=1).max().item()
    rho *= net['fc3.weight'].norm(dim=1).max().item()
    return rho

def eval_rho4(net,threshold):
    rho = 1
    v = net['conv1.weight_v']
    g = net['conv1.weight_g']
    conv1_weight = g * (v / v.norm())
    rho *= conv1_weight.norm().item()
    v = net['conv2.weight_v']
    g = net['conv2.weight_g']
    conv2_weight = g * (v / v.norm())
    rho *= conv2_weight.norm().item()
    # print(rho)
    rho *= rankedness_by_var(net['fc1.weight'], threshold=threshold)
    rho *=rankedness_by_var(net['fc2.weight'], threshold=threshold)
    rho *= rankedness_by_var(net['fc3.weight'], threshold=threshold)
    return rho

def eval_rho5(net):
    rho = 1
    v = net['conv1.weight_v']
    g = net['conv1.weight_g']
    conv1_weight = g * (v / v.norm())
    rho *= conv1_weight.norm().item()
    v = net['conv2.weight_v']
    g = net['conv2.weight_g']
    conv2_weight = g * (v / v.norm())
    rho *= conv2_weight.norm().item()
    rho *= rankedness_by_var(net['fc1.weight'], threshold=0.3)
    rho *= rankedness_by_var(net['fc2.weight'], threshold=0.3)
    rho *= rankedness_by_var(net['fc3.weight'], threshold=0.3)
    return rho

def bound(rho, net, training_loader, max_pixel_sums):
    n = len(training_loader.dataset)
    k = 10
    depth = 7
    degs = [5**2,3**2,5**2,3**2, 192*6*6, 384] #, 192]
    # degs = [5, 3, 5, 3, 1, 1, 1]
    delta = 0.001
    deg_prod = np.prod(degs)
    mult1 = (2 ** 1.5) * (rho + 1) / n
    mult2 = 1 + math.sqrt(2 * (np.log(2) * depth + np.log(deg_prod) + np.log(k)))
    mult3 = math.sqrt(max_pixel_sums * deg_prod)
    add1 = 3 * math.sqrt(np.log((2 * (rho + 2) ** 2) / delta) / (2 * n))
    bound = mult1 * mult2 * mult3 + add1
    return bound


def eval_rho1_poggio(net):
    rho = 1
    for weight_idx in [1, 2, 3, 4]:
        g = net[f'conv{weight_idx}.weight_g']
        v = net[f'conv{weight_idx}.weight_v']
        weight = g * (v / v.norm())
        rho *= weight.norm().item()
    rho *= net['fc.weight'].norm().max().item()
    return rho

def eval_rho2_poggio(net):
    rho = 1
    for weight_idx in [1, 2, 3, 4]:
        g = net[f'conv{weight_idx}.weight_g']
        v = net[f'conv{weight_idx}.weight_v']
        weight = g * (v / v.norm())
        rho *= weight.norm().item()
    rho *= net['fc.weight'].norm(dim=1).max().item()
    return rho

def eval_rho3_poggio(net):
    rho = 1
    for weight_idx in [1, 2, 3, 4]:
        g = net[f'conv{weight_idx}.weight_g']
        v = net[f'conv{weight_idx}.weight_v']
        weight = g * (v / v.norm())
        rho *= weight.norm(dim=(2, 3)).max().item()
    rho *= net['fc.weight'].norm(dim=1).max().item()
    return rho

def eval_rho4_poggio(net, threshold):
    rho = 1
    for weight_idx in [1, 2, 3, 4]:
        g = net[f'conv{weight_idx}.weight_g']
        v = net[f'conv{weight_idx}.weight_v']
        weight = g * (v / v.norm())
        rho *= weight.norm(dim=(2, 3)).max().item()
    rho *= rankedness_by_var(net['fc.weight'], threshold)
    return rho

def eval_rho5_poggio(net):
    rho = 1
    for weight_idx in [1, 2, 3, 4]:
        g = net[f'conv{weight_idx}.weight_g']
        v = net[f'conv{weight_idx}.weight_v']
        weight = g * (v / v.norm())
        rho *= weight.norm(dim=(2, 3)).max().item()
    rho *= rankedness_by_var(net['fc.weight'], 0.85)
    return rho


def bound_poggio(rho, net, training_loader, max_pixel_sums):
    n = len(training_loader.dataset)
    k = 10
    depth = 5
    degs = [2**2,2**2,2**2,2**2] #, 400]
    delta = 0.001
    deg_prod = np.prod(degs)
    mult1 = (2 ** 1.5) * (rho + 1) / n
    mult2 = 1 + math.sqrt(2 * (np.log(2) * depth + np.log(deg_prod) + np.log(k)))
    mult3 = math.sqrt(max_pixel_sums * deg_prod)
    add1 = 3 * math.sqrt(np.log((2 * (rho + 2) ** 2) / delta) / (2 * n))
    bound = mult1 * mult2 * mult3 + add1
    return bound

def calc_poggio_rankedness_stats(net, threshold):
    rho = 1
    for weight_idx in [1, 2, 3, 4]:
        g = net[f'conv{weight_idx}.weight_g']
        v = net[f'conv{weight_idx}.weight_v']
        weight = g * (v / v.norm())
        rho *= weight.norm(dim=(2, 3)).max().item()

    fc_rankedness = rankedness_by_var(net['fc.weight'], threshold=threshold)

    fc_frob = net['fc.weight'].norm().item()

    fc_frob_2 = net['fc.weight'].norm(dim=1).max().item()

    return rho, [fc_rankedness], [fc_frob], [fc_frob_2]

def calc_rankedness_stats(net, threshold, model_name="AlexNet"):
    if model_name == "PoggioNet":
        return calc_poggio_rankedness_stats(net, threshold)
    rho = 1
    v = net['conv1.weight_v']
    g = net['conv1.weight_g']
    conv1_weight = g * (v / v.norm())
    rho *= conv1_weight.norm().item()
    v = net['conv2.weight_v']
    g = net['conv2.weight_g']
    conv2_weight = g * (v / v.norm())
    rho *= conv2_weight.norm().item()

    fc1_rankedness = rankedness_by_var(net['fc1.weight'], threshold=threshold)
    fc2_rankedness = rankedness_by_var(net['fc2.weight'], threshold=threshold)
    fc3_rankedness = rankedness_by_var(net['fc3.weight'], threshold=threshold)

    fc1_frob = net['fc1.weight'].norm().item()
    fc2_frob = net['fc2.weight'].norm().item()
    fc3_frob = net['fc3.weight'].norm().item()


    fc1_frob_2 = net['fc1.weight'].norm(dim=1).max().item()
    fc2_frob_2 = net['fc2.weight'].norm(dim=1).max().item()
    fc3_frob_2 = net['fc3.weight'].norm(dim=1).max().item()

    return rho, [fc1_rankedness, fc2_rankedness, fc3_rankedness], [fc1_frob, fc2_frob, fc3_frob], [fc1_frob_2, fc2_frob_2, fc3_frob_2]

def calc_bounds(net, training_loader, max_pixel_sums, model_name="AlexNet", threshold=0.7):
    if model_name=="PoggioNet":
        rho1 = eval_rho1_poggio(net)
        rho2 = eval_rho2_poggio(net)
        rho3 = eval_rho3_poggio(net)
        rho4 = eval_rho4_poggio(net)
        # rho5 = eval_rho5_poggio(net)
        bound1 = bound_poggio(rho1, net, training_loader, max_pixel_sums)
        bound2 = bound_poggio(rho2, net, training_loader, max_pixel_sums)
        bound3 = bound_poggio(rho3, net, training_loader, max_pixel_sums)
        bound4 = bound_poggio(rho4, net, training_loader, max_pixel_sums, threshold=threshold)
        # bound5 = bound_poggio(rho5, net, training_loader, max_pixel_sums)

    else:
        rho1 = eval_rho1(net)
        rho2 = eval_rho2(net)
        rho3 = eval_rho3(net)
        rho4 = eval_rho4(net, threshold)
        # rho5 = eval_rho5(net)
        bound1 = bound(rho1, net, training_loader, max_pixel_sums)
        bound2 = bound(rho2, net, training_loader, max_pixel_sums)
        bound3 = bound(rho3, net, training_loader, max_pixel_sums)
        bound4 = bound(rho4, net, training_loader, max_pixel_sums)
        # bound5 = bound(rho5, net, training_loader, max_pixel_sums)
    return bound1, bound2, bound3, bound4, rho1, rho2, rho3, rho4