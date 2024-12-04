import torch.nn as nn
import torch
import torch.nn.utils.weight_norm as weight_norm


class PoggioNet(nn.Module):
    def __init__(self, num_input_channels=3, width=100, num_layers=3, num_output_classes=10, image_size=28):
        super(PoggioNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.width = width

        self.conv1 = weight_norm(nn.Conv2d(self.num_input_channels, self.width, kernel_size=2), dim=None)
        layers = [self.conv1]
        for layer in range(2, num_layers+1):
            setattr(self, f'conv{layer}', weight_norm(nn.Conv2d(self.width, self.width, kernel_size=2), dim=None))
            layers.append(getattr(self, f'conv{layer}'))
        
        self.layers = nn.Sequential(*layers)

        # Calculate final spatial dimensions after all convolutional layers
        final_size = image_size
        for _ in range(num_layers):
            final_size = (final_size - 2) // 1 + 1  # Kernel=2, Stride=1
        flattened_size = final_size * final_size * self.width  # output channels from conv4
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=flattened_size, out_features=num_output_classes)
        
        self.all_layers = [getattr(self, f'conv{layer}') for layer in range(1,num_layers+1)] + [self.fc]
        self.degs = [2 for _ in range(num_layers)]
        self.depth = num_layers + 1
        self.conv_depth = num_layers
        self.fc_depth = 1

    def forward(self, x):
        shapes = []
        for layer in range(1, self.conv_depth+1):
            x = getattr(self, f'conv{layer}')(x)
            shapes += [x.shape]
            x = nn.functional.relu(x)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)

        # Pass through fully connected layer and return the output
        x = self.fc(x)
        return x