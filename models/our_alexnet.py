import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # Unclear what out_channels should actually be in both modules
        # Unclear how to choose paramteres for LocalResponseNorm

        # ReLU is used in original alex net but not 100% specified in the paper
        # Dropout is used in original AlexNet but we aren't using it here as specified in the paper

        # Module 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # Overlapping pooling
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        
        # Module 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        
        # Fully connected layers (fc1 initialized later)
        self.fc1 = None  # Placeholder
        self.fc2 = nn.Linear(in_features=384, out_features=192)
        self.fc3 = nn.Linear(in_features=192, out_features=num_classes)
    
    def forward(self, x):
        # Module 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        
        # Module 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)

        # Dynamically initialize fc1 during the first forward pass
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 384).to(x.device)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x