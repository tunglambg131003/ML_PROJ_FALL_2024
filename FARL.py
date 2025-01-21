import timm
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Define the classifier module
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(input_dim // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
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

# Load FaRL16, FaRL64, or ResNeXt50 backbones
def load_backbone(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'FaRL64':
        # Load FaRL64 model
        farl_state = torch.load("/content/drive/MyDrive/ML Project Dataset /FaRL-Base-Patch16-LAIONFace20M-ep64.pth", map_location=device)
        model, _ = clip.load("ViT-B/16", device=device)
        model.load_state_dict(farl_state["state_dict"], strict=False)
        feature_dim = model.visual.output_dim  # CLIP feature dimension
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    model.head = nn.Identity()  # Remove classification head
    return model, feature_dim

# Compute precision, recall, F1 score, and accuracy
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    return acc, precision, recall, f1

# Main fine-tuning function
def fine_tune_and_evaluate(model_name, dataloaders, test_loader, num_classes, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, feature_dim = load_backbone(model_name)
    backbone = backbone.to(device).float()  # Ensure backbone is using float32 precision
    classifier = Classifier(input_dim=feature_dim, num_classes=num_classes).to(device).float()  # Ensure classifier is using float32 precision

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(backbone.parameters()) + list(classifier.parameters()), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5, last_epoch=-1)

    for epoch in range(num_epochs):
        backbone.eval()  # Freeze backbone weights
        classifier.train()

        running_loss = 0.0
        all_labels = []
        all_preds = []

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            with torch.no_grad():
                features = backbone.encode_image(inputs)  # Extract features from the image
            features = features.float()  # Ensure features are float32
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()

            # Store predictions and true labels for metrics calculation
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        # Compute training metrics
        train_acc, train_precision, train_recall, train_f1 = compute_metrics(all_labels, all_preds)

        # Print epoch results
        print(f"{model_name} - Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {running_loss / len(dataloaders['train']):.4f}, Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")

    # Evaluate on the test dataset after training
    classifier.eval()
    test_labels = []
    test_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            features = backbone.encode_image(inputs)
            features = features.float()
            outputs = classifier(features)

            _, predicted = torch.max(outputs, 1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())

    # Compute test metrics
    test_acc, test_precision, test_recall, test_f1 = compute_metrics(test_labels, test_preds)

    # Print test performance
    print(f"Test Performance for {model_name}")
    print(f"Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

# Example usage:
dataset_dir = 'ML Project Dataset /RGB-M'

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

# Number of classes
num_classes = len(full_train_dataset.classes)

# Fine-tune and evaluate FaRL16, FaRL64, and ResNeXt50
for model_name in ['FaRL64']:
    print(f"\n=== Fine-tuning and Evaluating {model_name} ===")
    fine_tune_and_evaluate(model_name, dataloaders, test_loader, num_classes, num_epochs=50)
