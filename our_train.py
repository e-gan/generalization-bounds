import yaml
from pathlib import Path
from data_utils import get_train_dataloader, get_test_dataloader
import os 
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from models.our_alexnet import AlexNet
from models.inception import InceptionNet
from models.poggio_net import PoggioNet
import numpy as np

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
def evaluate(model, valid_loader, loss_fn, num_classes, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #one hot encode labels for MSE loss
            if len(labels.shape) == 1:
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            loss += loss_fn(outputs, labels).item()

    accuracy = correct / total
    loss = loss / len(valid_loader)
    return loss, accuracy

def train(model, optimizer, loss_fn, lr_scheduler, reg_function, train_loader, valid_loader, num_epochs, save_dir, num_classes, device, save_epochs=10):
    last_5_train_accuracies = [0, 0, 0, 0, 0]
    #make model directory
    os.makedirs(save_dir, exist_ok=True)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        correct = 0
        total = 0
        torch.cuda.empty_cache()
        for step, (inputs, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #one hot encode labels for MSE loss
            if len(labels.shape) == 1:
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        
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
        
        train_accuracy = correct / total
            
        # evaluate
        val_loss, val_accuracy = evaluate(model, valid_loader, loss_fn, num_classes, device)
        wandb.log({"epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "train_accuracy": train_accuracy})

        #save model if train accuracy is not increasing
        # if last_5_train_accuracies:
        #     if train_accuracy <= np.mean(last_5_train_accuracies):
        #         torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pt"))
        #         last_5_train_accuracies = [] #don't save model again
        #     else:
        #         last_5_train_accuracies.pop(0)
        #         last_5_train_accuracies.append(train_accuracy)

        #save model every save_epochs
        if epoch % save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pt"))
    
    #save final model
    torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{num_epochs}.pt"))
    wandb.finish()

    #save model

def get_data_loaders(config):
    # should return two tensors, train_dataset and valid_dataset
    # should be normalized already
    train_loader = get_train_dataloader(dataset=config["data"]["dataset"],
                                        batch_size=config["training"]["batch_size"], 
                                        loss_fn = config["training"]["loss_fn"],
                                        corruption_type=config["data"]["corruption_type"],
                                        corruption_prob=config["data"]["corruption_prob"],
                                        num_classes=config["data"]["num_classes"],
                                        num_workers=config["data"]["num_workers"])
    
    test_loader = get_test_dataloader(dataset=config["data"]["dataset"],
                                        batch_size=config["training"]["batch_size"], 
                                        loss_fn = config["training"]["loss_fn"],
                                        num_classes=config["data"]["num_classes"],
                                        num_workers=config["data"]["num_workers"])
    
    image_size = test_loader.dataset[0][0].shape[-1]
    if image_size != config["data"]["image_size"]:
        raise ValueError(f"Image size mismatch. Model image size is {config['data']['image_size']} but dataset image size is {image_size}")

    return train_loader, test_loader

def get_loss_fn(config):
    loss_fn = config["training"]["loss_fn"]
    if loss_fn == 'CE':
        return nn.CrossEntropyLoss()
    elif loss_fn == 'MSE':
        return nn.MSELoss()
    elif loss_fn == 'l1':
        return nn.L1Loss()

def get_lr_scheduler(config, optimizer, step_size=1, gamma=0.95):
    #NOTE: They have a decay factor of 0.95 every epoch
    # They use an initial learning rate of 0.1 for Inception and 0.01 for AlexNet

    lr_scheduler = config["training"]["lr_scheduler"]
    if lr_scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif lr_scheduler == None:
        return None
    
def get_optimizer(config, model):
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
    elif config["model"]["name"] == "PoggioNet":
        return PoggioNet(
                        width=config["model"]["width"], 
                        num_layers=config["model"]["num_layers"], 
                        num_output_classes=config["data"]["num_classes"], 
                        image_size=config["data"]["image_size"])  
                    
    
@hydra.main(config_path="configs", config_name="train_config")
def main(config: DictConfig):
    """
    Main training function. Hydra automatically loads the configuration into `config`.
    """

    wandb.login(key="5aee75a09d43e7f6c9ec80e003687a8a3a820b08")

    config = OmegaConf.to_container(config, resolve=True)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    step_size = config["training"]["lr_scheduler_params"]["step_size"]
    gamma = config["training"]["lr_scheduler_params"]["gamma"]

    train_loader, valid_loader = get_data_loaders(config)

    #reset seed so we can get different models
    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = get_model(config).to(device)
    print(model)
    loss_fn = get_loss_fn(config)
    reg_function = get_regularizer(config)
    optimizer = get_optimizer(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer, step_size, gamma)

    num_epochs = config["training"]["num_epochs"]
    save_epochs = config["training"]["save_epochs"]
    num_classes = config["data"]["num_classes"]

    wandb.init(project="generalization_bounds", config=config)
    run_name = wandb.run.name
    print(f"Run name: {run_name}")

    model_name = config["model"]["name"]
    if not config["data"]["corruption_type"] == "None":
        type = config["data"]["corruption_type"]
        prob = config["data"]["corruption_prob"]
        save_dir = os.path.join(script_dir, f"saved_models/{model_name}/{type}/{prob}/{run_name}")
    else:
        save_dir = os.path.join(script_dir, f"saved_models/{model_name}/regular/{run_name}")

    train(model, optimizer, loss_fn, lr_scheduler, reg_function, train_loader, valid_loader, num_epochs, save_dir, num_classes, device, save_epochs)


if __name__ == "__main__":
    #set seed
    main()