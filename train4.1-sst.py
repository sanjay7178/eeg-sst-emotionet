import pandas as pd
import mne
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import random
from models.cnn.sst_emotion_net import SSTEmotionNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR

# Initialize wandb
wandb.init(project="EEG_classification", name="SSTEmotionNet_training")

# Read the CSV file
df = pd.read_csv("new_update_final_output.csv")

# Initialize lists to store EEG data and labels
eeg_data_list = []
labels = []

# Iterate over the file paths in the CSV
for idx, row in df.iterrows():
    file_path = row["file_path"]
    label = row["label"]
    # Load the .fif file using MNE
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    # Extract the EEG data as a NumPy array
    data = raw.get_data()
    eeg_data_list.append(data)
    labels.append(label)

# Find the maximum length of the EEG data arrays
max_length = max(data.shape[1] for data in eeg_data_list)

# Pad or truncate the EEG data arrays to ensure they all have the same shape
eeg_data_array = np.array(
    [
        np.pad(data, ((0, 0), (0, max_length - data.shape[1])), mode="constant")
        if data.shape[1] < max_length
        else data[:, :max_length]
        for data in eeg_data_list
    ]
)

# Convert labels to a NumPy array
labels = np.array(labels)

# Binarize labels for AUROC calculation
unique_labels = np.unique(labels)
labels_binarized = label_binarize(labels, classes=unique_labels)

# Split the data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(
    eeg_data_array, labels, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42
)
# Now, 60% train, 20% validation, 20% test

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Update model class
# Remove EEGNet class and keep only SSTModelWrapper
class SSTModelWrapper(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SSTModelWrapper, self).__init__()
        # Calculate spectral and temporal channels
        total_channels = input_shape[1]
        spectral_channels = total_channels // 5  # Using 1/5 of channels for spectral
        temporal_channels = total_channels - spectral_channels

        self.sst = SSTEmotionNet(
            grid_size=(16, 16),  # Resize input to 16x16
            spectral_in_channels=spectral_channels,
            temporal_in_channels=temporal_channels,
            num_classes=num_classes,
            densenet_dropout=0.5,
            task_dropout=0.3,
        )

    def forward(self, x):
        return self.sst(x)


# Update model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = X_train_tensor.shape
num_classes = len(np.unique(labels))
model = SSTModelWrapper(input_shape, num_classes)
model = model.to(device)


# Update data preprocessing
# Add resize transform to the EEGDataset
class EEGDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = data
        # Add normalization
        mean = torch.mean(data, dim=(0,2))
        std = torch.std(data, dim=(0,2))
        self.data = (data - mean.unsqueeze(0).unsqueeze(-1)) / (std.unsqueeze(0).unsqueeze(-1) + 1e-8)
        self.labels = labels
        self.augment = augment

    def __getitem__(self, idx):
        data = self.data[idx]  # Shape: [channels, time]

        # Calculate the closest perfect square size that can fit the data
        time_points = data.shape[1]
        grid_size = int(np.ceil(np.sqrt(time_points)))

        # Pad the data to make it fit into a square grid
        pad_size = grid_size * grid_size - time_points
        if pad_size > 0:
            data = F.pad(data, (0, pad_size), "constant", 0)

        # Reshape into grid
        data = data.reshape(data.shape[0], grid_size, grid_size)

        # Add batch dimension for interpolate
        data = data.unsqueeze(0)  # [1, C, H, W]

        # Resize to 16x16 using interpolation
        data = F.interpolate(data, size=(16, 16), mode="bilinear", align_corners=True)

        # Remove batch dimension
        data = data.squeeze(0)  # [C, 16, 16]

        if self.augment:
            data = self.augment_data(data.numpy())
            data = torch.from_numpy(data)

        return data, self.labels[idx]

    def __len__(self):
        return len(self.data)

    def augment_data(self, data):
        # Apply random augmentation techniques
        augmentation_methods = [
            self.add_gaussian_noise,
            self.time_shift,
            self.scale_signals,
            self.frequency_shift,
        ]
        # Randomly choose which augmentations to apply
        num_augmentations = random.randint(1, len(augmentation_methods))
        augmentations = random.sample(augmentation_methods, num_augmentations)
        for augmentation in augmentations:
            data = augmentation(data)
        return data.astype(np.float32)  # Ensure data is float32

    def add_gaussian_noise(self, data):
        noise = np.random.normal(0, 0.01, data.shape)
        data_noisy = data + noise
        return data_noisy

    def time_shift(self, data):
        max_shift = data.shape[1] // 10  # Shift up to 10% of the signal length
        shift = np.random.randint(-max_shift, max_shift)
        data_shifted = np.roll(data, shift, axis=1)
        return data_shifted

    def scale_signals(self, data):
        scaling_factor = np.random.uniform(0.9, 1.1)
        data_scaled = data * scaling_factor
        return data_scaled

    def frequency_shift(self, data):
        # Apply Fourier transform
        freq_data = np.fft.fft(data, axis=1)
        # Shift frequencies
        shift = np.random.randint(-5, 5)
        freq_data_shifted = np.roll(freq_data, shift, axis=1)
        # Apply inverse Fourier transform
        data_shifted = np.fft.ifft(freq_data_shifted, axis=1)
        # Keep only the real part and ensure float32
        data_shifted = np.real(data_shifted).astype(np.float32)
        return data_shifted


# Update batch size for memory efficiency
batch_size = 64

# Create dataloaders with the updated dataset
train_dataset = EEGDataset(X_train_tensor, y_train_tensor, augment=True)
val_dataset = EEGDataset(X_val_tensor, y_val_tensor, augment=False)
test_dataset = EEGDataset(X_test_tensor, y_test_tensor, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# class EEGNet(nn.Module):
#     def __init__(self, input_shape, num_classes):
#         super(EEGNet, self).__init__()
#         # Input shape will be [channels, 16, 16] after preprocessing
#         self.channels = input_shape[1]
#         self.input_size = 16 * 16  # Since we resize to 16x16

#         self.fc1 = nn.Linear(self.channels * self.input_size, 128)
#         self.relu = nn.ReLU()
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(128, 64)
#         self.dropout2 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         # x shape: [batch, channels, 16, 16]
#         x = x.view(x.size(0), -1)  # Flatten: [batch, channels * 16 * 16]
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x


# Update training configuration
criterion = nn.CrossEntropyLoss()
# 3. Update optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10, 
    T_mult=2,
    eta_min=1e-6
)

# Number of epochs
epochs = 500

# Initialize warmup scheduler
warmup_epochs = 5
warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.1,
    end_factor=1.0, 
    total_iters=warmup_epochs
)

# Replace ReduceLROnPlateau with CosineAnnealingLR
main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs - warmup_epochs,
    eta_min=1e-6
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[warmup_epochs]
)

# Number of epochs
epochs = 500

best_val_accuracy = 0.0
best_model_path = "best_model.pth"

# Gradient clipping
max_grad_norm = 1.0

# Update the scheduler steps in the training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Update warmup scheduler per iteration during warmup
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    # Add gradient norm monitoring
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item()
    
    wandb.log({
        "gradient_norm": grad_norm,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    # Validation
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        val_all_targets = []
        val_all_predictions = []
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
            val_all_targets.extend(target.cpu().numpy())
            val_all_predictions.extend(predicted.cpu().numpy())
        val_accuracy = 100 * val_correct / val_total
        # Compute validation metrics
        val_precision = precision_score(
            val_all_targets, val_all_predictions, average="macro",zero_division=0
        )
        val_recall = recall_score(val_all_targets, val_all_predictions, average="macro")
        val_auroc = roc_auc_score(
            label_binarize(val_all_targets, classes=unique_labels),
            label_binarize(val_all_predictions, classes=unique_labels),
            average="macro",
            multi_class="ovo",
        )
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    # Update main scheduler per epoch after warmup
    if epoch >= warmup_epochs:
        scheduler.step()  # Remove val_accuracy parameter

    train_accuracy = 100 * correct / total
    # Compute metrics
    train_precision = precision_score(
        all_targets, all_predictions, average="macro", zero_division=0
    )

    train_recall = recall_score(all_targets, all_predictions, average="macro")
    train_auroc = roc_auc_score(
        label_binarize(all_targets, classes=unique_labels),
        label_binarize(all_predictions, classes=unique_labels),
        average="macro",
        multi_class="ovo",
    )
    # Validation
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        val_all_targets = []
        val_all_predictions = []
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
            val_all_targets.extend(target.cpu().numpy())
            val_all_predictions.extend(predicted.cpu().numpy())
        val_accuracy = 100 * val_correct / val_total
        # Compute validation metrics
        val_precision = precision_score(
            val_all_targets, val_all_predictions, average="macro",zero_division=0
        )
        val_recall = recall_score(val_all_targets, val_all_predictions, average="macro")
        val_auroc = roc_auc_score(
            label_binarize(val_all_targets, classes=unique_labels),
            label_binarize(val_all_predictions, classes=unique_labels),
            average="macro",
            multi_class="ovo",
        )
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
    # Log metrics to wandb
    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "train_accuracy": train_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_auroc": train_auroc,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_auroc": val_auroc,
        }
    )
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, "
        f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%"
    )

# Load best model for testing
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    test_accuracy = 100 * correct / total
    # Compute test metrics
    test_precision = precision_score(all_targets, all_predictions, average="macro",zero_division=0)
    test_recall = recall_score(all_targets, all_predictions, average="macro")
    test_auroc = roc_auc_score(
        label_binarize(all_targets, classes=unique_labels),
        label_binarize(all_predictions, classes=unique_labels),
        average="macro",
        multi_class="ovo",
    )
    # Log test metrics to wandb
    wandb.log(
        {
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_auroc": test_auroc,
        }
    )
    print(f"Test Accuracy: {test_accuracy:.2f}%")

# Finish wandb run
wandb.finish()
