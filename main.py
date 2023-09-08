import os
import logging

import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from tqdm import tqdm

log_dir = 'logs'  # Specify the directory where logs will be saved
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')


# Define a function to log information
def log_info(message):
    logging.info(message)
    print(message)


# Set the device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet-50 model
model = resnet50(pretrained=True)
num_classes = 1081  # plantnet has 1,081 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
