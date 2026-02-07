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
    def __init__(self, board_w, board_h, hiddem_size=256, output_size=3):
        super().__init__()
        # Input shape: (3, board_h, board_w)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size dynamically
        # For board 800x600 with BLOCK_SIZE 20, w=40, h=30
        self.flat_features = 64 * board_w * board_h

        self.fc1 = nn.Linear(self.flat_features, hiddem_size)
        self.fc2 = nn.Linear(hiddem_size, output_size)

    def forward(self, x):
        # x shape: (Batch, 3, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # FIX: Flatten everything after the batch dimension
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
