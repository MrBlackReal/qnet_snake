# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def save(self, file_name="model.pth"):
        model_folder_path = "./models"

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path = os.path.join(model_folder_path, file_name)

        print("Saving:", path)
        torch.save(self.state_dict(), path)

    def load(self, file_name="model.pth"):
        file_path = os.path.join("./models", file_name)

        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            self.eval()

            print("Loaded:", file_name)
            return True

        print("Failed to load:", file_name)
        return False


class Conv_QNet(nn.Module):
    def __init__(self, board_w, board_h, hidden_size=256, output_size=3):
        super().__init__()
        # Input shape: (3, board_h, board_w)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size dynamically
        # Create a dummy input to pass through the convolutional layers
        # to get the exact output size.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, board_h, board_w)
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            self.flat_features = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flat_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (Batch, 3, H, W)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        
        return self.fc2(x)

    def save(self, file_name="model.pth"):
        model_folder_path = "./models"

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path = os.path.join(model_folder_path, file_name)

        print("Saving:", path)
        torch.save(self.state_dict(), path)

    def load(self, file_name="model.pth"):
        file_path = os.path.join("./models", file_name)

        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            self.eval()

            print("Loaded:", file_name)
            return True

        print("Failed to load:", file_name)
        return False
