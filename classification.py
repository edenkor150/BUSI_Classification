import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
import random

from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import Precision, Recall

RANDOM_SEED = 159

# Paths
folder_path = r'E:\Eden\US_images'

# Image transformation
train_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.Grayscale(),
    # transforms.RandomAutocontrast(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print('Round')
# # Create Dataset using ImageFolder
# dataset_train = ImageFolder(
#     'cloud-type-classification2/images/clouds_train',
#     transform=train_transforms,
# )
#
# dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Define feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        # Define classifier
        self.fc1 = nn.Linear(16*56*56, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, num_classes)

    def forward(self, x):
        # Pass input through feature extractor and classifiers
        x = self.feature_extractor(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.log_softmax(self.fc4(x),dim=1)
        return x

def validate(model, dataloader, criter, dev):
    """
    Evaluate model on validation/test data
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    # Initialize metrics
    metric_precision = Precision(task="binary", average="macro")
    metric_recall = Recall(task="binary", average="macro")

    with torch.no_grad():  # Disable gradient computation
        for images, labels in dataloader:
            images = images.to(dev)
            labels = labels.to(dev)

            # Forward pass
            outputs = model(images)
            loss = criter(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update metrics
            metric_precision(predicted, labels)
            metric_recall(predicted, labels)

            total_loss += loss.item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    # Compute precision and recall
    precision = metric_precision.compute()
    recall = metric_recall.compute()

    return avg_loss, accuracy, precision, recall

if __name__ == "__main__":
    # Create Dataset using ImageFolder
    # Create datasets for all three splits
    dataset_train = ImageFolder(folder_path, transform=train_transforms)
    X_train, X_val, y_train, y_val = train_test_split(dataset_train, test_size=0.2, random_state=RANDOM_SEED, stratify=dataset_train)
    X_train, X_test, y_train, y_test = train_test_split(X_train, test_size=0.3, random_state=RANDOM_SEED, stratify=dataset_train)
    dataset_val = ImageFolder(val_folder_path, transform=val_transforms)

    # Create dataloaders
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Training configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    # Training variables
    num_epochs = 30
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Validation phase
        val_loss, val_accuracy, val_precision, val_recall = validate(net, val_loader, criterion, device)

        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # # Learning rate scheduling
        # scheduler.step(val_loss)

        # Early stopping check
        if np.round(val_loss,4) < np.round(best_val_loss,4):
            best_val_loss = val_loss
            best_model_state = net.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    # Load best model
    net.load_state_dict(best_model_state)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # # Define metrics
    # metric_precision = Precision(task="multiclass", num_classes=7, average="macro")
    # metric_recall = Recall(task="multiclass", num_classes=7, average="macro")

    # # Define precision metric
    # metric_precision = Precision(
    #     task="multiclass", num_classes=7, average=None
    # )
    # metric_recall = Recall(
    #     task="multiclass", num_classes=7, average=None
    # )
    #
    # net.eval()
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         outputs = net(images)
    #         _, preds = torch.max(outputs, 1)
    #         metric_precision(preds, labels)
    #         metric_recall(preds, labels)
    # precision = metric_precision.compute()
    # recall = metric_recall.compute()
    #
    # # Get precision per class
    # precision_per_class = {
    #     k: precision[v].item()
    #     for k, v
    #     in dataset_test.class_to_idx.items()
    # }
    # recall_per_class = {
    #     k: recall[v].item()
    #     for k, v
    #     in dataset_test.class_to_idx.items()
    # }
    # print(precision_per_class)
    # print(recall_per_class)

# # Define the model
# net = Net(num_classes=7)
# # Define the loss function
# criterion = nn.CrossEntropyLoss()
# # Define the optimizer
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
#
# epoch_loss = []
# count=0
# for epoch in range(30):
#     running_loss = 0.0
#     # Iterate over training batches
#     for images, labels in dataloader_train:
#         optimizer.zero_grad()
#         outputs = net(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     epoch_loss.append(running_loss / len(dataloader_train))
#     print(f"Epoch {epoch + 1}, Loss: {epoch_loss[epoch]:.4f}")
#
#     scheduler.step(epoch_loss[epoch])
#
#     if epoch_loss[epoch] > np.min(epoch_loss):
#         count+=1
#     if count==3:
#         break
#
#
# test_transforms = transforms.Compose([
#     # Add horizontal flip and rotation
#     transforms.ToTensor(),
#     transforms.Resize((64, 64)),
# ])
#
# # Create Dataset using ImageFolder
# dataset_test = ImageFolder(
#     'cloud-type-classification2/images/clouds_test',
#     transform=test_transforms,
# )
#
# dataloader_test = DataLoader(
#   dataset_test, shuffle=True, batch_size=1
# )
#
# # # Define metrics
# # metric_precision = Precision(task="multiclass", num_classes=7, average="macro")
# # metric_recall = Recall(task="multiclass", num_classes=7, average="macro")
#
# # Define precision metric
# metric_precision = Precision(
#     task="multiclass", num_classes=7, average=None
# )
# metric_recall = Recall(
#     task="multiclass", num_classes=7, average=None
# )
#
# net.eval()
# with torch.no_grad():
#     for images, labels in dataloader_test:
#         outputs = net(images)
#         _, preds = torch.max(outputs, 1)
#         metric_precision(preds, labels)
#         metric_recall(preds, labels)
# precision = metric_precision.compute()
# recall = metric_recall.compute()
#
# # Get precision per class
# precision_per_class = {
#     k: precision[v].item()
#     for k, v
#     in dataset_test.class_to_idx.items()
# }
# recall_per_class = {
#     k: recall[v].item()
#     for k, v
#     in dataset_test.class_to_idx.items()
# }
# print(precision_per_class)
# print(recall_per_class)
