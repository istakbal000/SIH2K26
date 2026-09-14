import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cnn_model import get_fire_cnn_model
from src.image_preprocessing import get_dataloaders
from src.config import get_project_root, config

def train_image_model(num_epochs=20, batch_size=32, learning_rate=0.001):
    print("=== Training Forest Fire Image Classification Model (CNN) ===")
    
    raw_data_dir = get_project_root() / config['data']['raw_dir']
    
    # We check if forest_fire/industrial_fire folders exist
    if not (raw_data_dir / 'forest_fire').exists() or not (raw_data_dir / 'industrial_fire').exists():
        print(f"ERROR: Image folders 'forest_fire' and/or 'industrial_fire' not found in {raw_data_dir}")
        return
        
    print("Setting up DataLoaders...")
    train_loader, val_loader = get_dataloaders(raw_data_dir, batch_size=batch_size)
    
    if train_loader is None:
        print("ERROR: Failed to load image dataset.")
        return
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = get_fire_cnn_model(pretrained=True).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    # Only optimize parameters that require gradients (the fc layer, since we froze the rest)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_model_weights = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            preds = torch.sigmoid(outputs) >= 0.5
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                
                preds = torch.sigmoid(outputs) >= 0.5
                val_running_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_weights = model.state_dict()
            
    # Save the best model
    models_dir = get_project_root() / config['output']['models']
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'forest_fire_image_model.pth'
    
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining Complete. Best model saved to {model_path}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_image_model()
