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

import data

log_dir = '/scratch/ssd004/scratch/kkasa/results/aum_test'  # Specify the directory where logs will be saved
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')


# Define a function to log information
def log_info(message):
    logging.info(message)
    print(message)


data_dir = '/h/kkasa/datasets/plantnet-300k/images/'
batch_size = 256
image_size = 256
crop_size = 224

train_loader, val_loader, test_loader, dataset_attributes = data.get_plantnet_data(root=data_dir, image_size=image_size,
                                                                                   crop_size=crop_size,
                                                                                   batch_size=batch_size, num_workers=4,
                                                                                   pretrained=True)

# Set the device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet-50 model
model = resnet50(pretrained=True)
num_classes = 1081  # plantnet has 1,081 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Send the model to the GPU if available
model = model.to(device)
# Wrap the model with DataParallel to use multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    log_info(f'Using {torch.cuda.device_count()} GPUs')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    val_total = 0
    val_correct = 0

    # go through train set
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Validation'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / train_total
    epoch_train_acc = train_correct / train_total
    epoch_val_acc = val_correct / val_total

    # Save the loss and accuracy at the end of each epoch
    log_info(
        f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f} Train Acc: {epoch_train_acc:.4f} Val Acc: {epoch_val_acc:.4f}')

# go through test data
test_total = 0
test_correct = 0
model.eval()
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc=f'Testing'):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (preds == labels).sum().item()

test_acc = test_correct / test_total
log_info(f'Test Accuracy: {test_acc}')
