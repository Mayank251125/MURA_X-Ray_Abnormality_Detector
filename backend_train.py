import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import kagglehub

# ==========================================
# 1. Dataset Download & Configuration
# ==========================================
print("[Backend] Initializing KaggleHub download...")
try:
    # Download the dataset files (Images) instead of just the CSV
    # This returns the local path where files were downloaded
    dataset_path = kagglehub.dataset_download("cjinny/mura-v11")
    print(f"[Backend] Dataset downloaded to: {dataset_path}")
    
    # The dataset usually extracts to a folder. We need to find 'MURA-v1.1'
    # Check if MURA-v1.1 is inside the download path
    possible_dir = os.path.join(dataset_path, "MURA-v1.1")
    if os.path.exists(possible_dir):
        DATA_DIR = possible_dir
    else:
        DATA_DIR = dataset_path # Fallback if structure is flat
        
    print(f"[Backend] Training Data Directory set to: {DATA_DIR}")

except Exception as e:
    print(f"[Error] Failed to download dataset: {e}")
    # Fallback for manual path if download fails
    DATA_DIR = "./MURA-v1.1" 

MODEL_SAVE_PATH = "mura_resnet18.pth"
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Model Definition
# ==========================================
def get_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

# ==========================================
# 3. Dataset Loader
# ==========================================
class MuraDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        print(f"[Backend] Scanning files in {self.root_dir}...")
        
        if not os.path.exists(self.root_dir):
            print(f"[Warning] Split directory not found: {self.root_dir}")
            return

        # Walk through directory structure: BodyPart -> Patient -> Study
        for body_part in os.listdir(self.root_dir):
            body_part_path = os.path.join(self.root_dir, body_part)
            if not os.path.isdir(body_part_path): continue
            
            # Use tqdm here if there are many folders, but standard loop is cleaner for logs
            for patient in os.listdir(body_part_path):
                patient_path = os.path.join(body_part_path, patient)
                if not os.path.isdir(patient_path): continue

                for study in os.listdir(patient_path):
                    study_path = os.path.join(patient_path, study)
                    if not os.path.isdir(study_path): continue
                    
                    # MURA Logic: 'positive' in folder name = Abnormal (1)
                    label = 1 if 'positive' in study else 0
                    
                    # Find images (png, jpg, etc)
                    images = glob.glob(os.path.join(study_path, "*.png")) + \
                             glob.glob(os.path.join(study_path, "*.jpg"))
                             
                    for img_file in images:
                        self.image_paths.append(img_file)
                        self.labels.append(label)

        print(f"[Backend] Found {len(self.image_paths)} images for {split}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.float32)
            
        return image, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 4. Training Function
# ==========================================
def train_model():
    print(f"[Backend] Using device: {DEVICE}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        train_dataset = MuraDataset(DATA_DIR, split='train', transform=transform)
        valid_dataset = MuraDataset(DATA_DIR, split='valid', transform=transform)
        
        if len(train_dataset) == 0:
            print("[Error] No training data found. Check the download path.")
            return

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"[Error] Failed to create DataLoaders: {e}")
        return

    model = get_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    print(f"\n[Backend] Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
            
            loop.set_postfix(loss=loss.item())

        train_acc = correct_preds / total_preds
        avg_loss = running_loss / len(train_loader)
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(avg_loss)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                labels = labels.unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(valid_loader)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[Backend] Model saved to {MODEL_SAVE_PATH}")
    
    # Plot Results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label="Train Acc")
    plt.plot(history['val_acc'], label="Valid Acc")
    plt.title("Accuracy")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Valid Loss")
    plt.title("Loss")
    plt.legend()
    
    plt.savefig("training_results.png")
    print("[Backend] Results saved to training_results.png")

if __name__ == "__main__":
    train_model()


