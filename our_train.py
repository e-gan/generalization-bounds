import yaml
from pathlib import Path
import os 
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from models.our_alexnet import AlexNet
from models.inception import InceptionNet
import os

import wandb

try:
    # Use __file__ if available
    script_dir = Path(__file__).resolve().parent
except NameError:
    # Fallback to current working directory if __file__ is not available
    # probably in jupyter notebook
    script_dir = Path(".").resolve()

# Step 1: Load the config.yaml file
def load_config():
    config_path = os.path.join(script_dir, "configs/train_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Evaluation function
def evaluate(model, valid_loader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            if labels.size(1) > 1:
                #labels are onehot encoded
                labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss = loss / len(valid_loader)
    return loss, accuracy

def train(model, optimizer, loss_fn, lr_scheduler, reg_function, train_loader, valid_loader, num_epochs, run_name, save_epochs=10):
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        torch.cuda.empty_cache()
        for step, (inputs, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            
            # forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            if reg_function is not None:
                loss += reg_function(model)
            loss.backward()
            optimizer.step()

            wandb.log({"step": step + len(train_loader) * epoch,
                        "train_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr']})
            
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        # evaluate
        val_loss, val_accuracy = evaluate(model, valid_loader, loss_fn, device)
        wandb.log({"epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy})

        #save model
        if save_epochs != 0 and epoch % save_epochs == 0:
            continue
            #save logic

    wandb.finish()

    #save model

def get_data_loaders(config):

    ################################################
    ##### get_datasets needs to be implemented #####
    ################################################

    # should return two tensors, train_dataset and valid_dataset
    # should be normalized already
    # train_dataset, valid_dataset = get_datasets(config["data"])

    #then delete this:
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    loss_fn = config["training"]["loss_fn"]
    #need to onehot encode labels for MSE loss    
    if loss_fn == 'MSE':
        train_dataset.targets = torch.nn.functional.one_hot(torch.tensor(train_dataset.targets), num_classes=config["data"]["num_classes"]).float()
        valid_dataset.targets = torch.nn.functional.one_hot(torch.tensor(valid_dataset.targets), num_classes=config["data"]["num_classes"]).float()

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
        
    return train_loader, valid_loader

def get_loss_fn(config):
    loss_fn = config["training"]["loss_fn"]
    if loss_fn == 'CE':
        return nn.CrossEntropyLoss()
    elif loss_fn == 'MSE':
        return nn.MSELoss()
    elif loss_fn == 'l1':
        return nn.L1Loss()

def get_lr_scheduler(config):
    #NOTE: "we do not need to change the learning rate schedule [for fitting random labels]" (Zhang et al., 2017)
    # however, for regular labels they have a decay factor of 0.95 every epoch
    # They use an initial learning rate of 0.1 for Inception and 0.01 for AlexNet

    lr_scheduler = config["training"]["lr_scheduler"]
    if lr_scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    elif lr_scheduler == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif lr_scheduler == None:
        return None
    
def get_optimizer(config):
    optimizer = config["training"]["optimizer"]
    lr = config["training"]["learning_rate"]
    
    if optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=config["training"]["momentum"])
    elif optimizer == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=lr)

def L2_reg(weight_decay):
    def reg(model):
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg * weight_decay
    return reg

def L1_reg(weight_decay):
    def reg(model):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, p=1)
        return l1_reg * weight_decay
    return reg

def get_regularizer(config):
    reg = config["training"]["regularization"]
    if reg == 'L1':
        return L1_reg(config["training"]["weight_decay"])
    elif reg == 'L2':
        return L2_reg(config["training"]["weight_decay"])
    elif reg == None:
        return None

def get_model(config):
    if config["model"]["name"] == "AlexNet":
        return AlexNet(num_classes=config["data"]["num_classes"])
    elif config["model"]["name"] == "InceptionNet":
        return InceptionNet(num_classes=config["data"]["num_classes"])

if __name__ == "__main__":
    config = load_config()

    wandb.login(key="your_api_key_here")

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader = get_data_loaders(config)
    model = get_model(config).to(device)
    loss_fn = get_loss_fn(config)
    lr_scheduler = get_lr_scheduler(config)
    reg_function = get_regularizer(config)
    optimizer = get_optimizer(config)

    num_epochs = config["training"]["num_epochs"]
    save_epochs = config["training"]["save_epochs"]

    wandb.init(project="generalization_bounds", config=config)
    run_name = wandb.run.name

    train(model, optimizer, loss_fn, lr_scheduler, reg_function, train_loader, valid_loader, num_epochs, run_name, save_epochs)