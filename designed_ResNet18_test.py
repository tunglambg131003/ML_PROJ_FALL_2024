import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Define the classifier module
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class TNN_FACE_Drop(nn.Module):
    def __init__(self):
        super(TNN_FACE_Drop, self).__init__()
        self.cnn1a = nn.Conv2d(3, 16, kernel_size=5, stride=1)
        self.cnn2a = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.cnn3a = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1)
        self.drop2d = nn.Dropout(p=0.2)
        self.linear = nn.Linear(12544, 4)

    def forward(self, x):
        x = torch.relu(self.cnn1a(x))
        x = torch.relu(self.cnn2a(x))
        x = torch.relu(self.cnn3a(x))
        x = self.pooling(x)
        x = self.drop2d(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# Data augmentation and preprocessing pipeline
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_resnet18_backbone(num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    feature_dim = model.fc.in_features
    model.fc = Classifier(input_dim=feature_dim, num_classes=num_classes)
    return model.to(device), feature_dim

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    return acc, precision, recall, f1

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        scheduler.step()

        # Compute training metrics
        train_acc, train_precision, train_recall, train_f1 = compute_metrics(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {running_loss / len(dataloaders['train']):.4f}, Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")

# Example usage with ResNet-18 and Triplet Loss
def fine_tune_with_triplet_loss(dataset_dir, num_epochs=50):
    # Load datasets
    full_train_dataset = datasets.ImageFolder(root=dataset_dir + '/train', transform=transform)
    test_dataset = datasets.ImageFolder(root=dataset_dir + '/test', transform=transform)

    # Split the training dataset
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    dataloaders = {'train': train_loader, 'val': val_loader}

    num_classes = len(full_train_dataset.classes)

    model, _ = load_resnet18_backbone(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5, last_epoch=-1)

    train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs)

    # Evaluate on test set
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_labels = []
    test_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())

    test_acc, test_precision, test_recall, test_f1 = compute_metrics(test_labels, test_preds)
    print(f"\nTest Performance:")
    print(f"Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

# Main execution
dataset_dir = 'ML Project/RGB-M'
fine_tune_with_triplet_loss(dataset_dir)
