import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm


class AlexNet(nn.Module):
    # settings is a dict containing the following keys:
    # - num_input_channels: 3
    # - num_output_classes: 10
    # - dropout: float, dropout rate
    def __init__(self, settings) -> None:
        super(AlexNet).__init__()
        self.degs = [11,3,5,3,3,3,3,3]
        self.conv_depth = 8
        self.fc_depth = 3

        self.conv1 = weight_norm(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = weight_norm(nn.Conv2d(64, 192, kernel_size=5, padding=2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = weight_norm(nn.Conv2d(192, 384, kernel_size=3, padding=1))
        self.conv4 = weight_norm(nn.Conv2d(384, 256, kernel_size=3, padding=1))
        self.conv5 = weight_norm(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout1 = nn.Dropout(p=settings.dropout)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.dropout2 = nn.Dropout(p=settings.dropout)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, settings.num_output_classes)

        layers = [self.conv1,self.maxpool1, self.conv2, self.maxpool2, self.conv3, self.conv4, self.conv5, self.maxpool3, self.avgpool]
        self.layers = nn.Sequential(*layers)
        self.all_layers = [self.conv1,self.maxpool1, self.conv2, self.maxpool2, self.conv3, self.conv4, self.conv5, self.maxpool3, self.avgpool, self.fc1, self.fc2, self.fc3]

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=settings.dropout),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=settings.dropout),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 10),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shapes =[x.shape]
        x = self.conv1(x)
        shapes += [x.shape]
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        shapes += [x.shape]
        x = self.conv2(x)
        shapes += [x.shape]
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        shapes += [x.shape]
        x = self.conv3(x)
        shapes += [x.shape]
        x = nn.functional.relu(x)
        x = self.conv4(x)
        shapes += [x.shape]
        x = nn.functional.relu(x)
        x = self.conv5(x)
        shapes += [x.shape]
        x = nn.functional.relu(x)
        x = self.maxpool3(x)
        shapes += [x.shape]
        assert x.shape[2] == 6 and x.shape[3] == 6
        x = self.avgpool(x)
        shapes += [x.shape]
        

        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x, shapes
    
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x
    

def alexnet(settings):
    return AlexNet(settings)