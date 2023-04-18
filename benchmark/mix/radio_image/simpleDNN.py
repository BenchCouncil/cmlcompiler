import torch 
import torch.nn as nn
import torch.nn.functional as F

class simpleDNN(torch.nn.Module):
    """
    simpleDNN to extract features from radio image
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5) 
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(6400, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, num_classes)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        #x = F.softmax(self.fc4(x), dim=-1)
        x = torch.sigmoid(self.fc4(x))
        return x

class simple_feature(torch.nn.Module):
    """
    simple feature model
    a submodule of simpleDNN
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5) 
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(6400, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x