import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# pyrefly: ignore [missing-import]
from torchvision import transforms
from pathlib import Path

class FireImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images. Should contain 'fire' and 'nofire' subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 1.0 = Forest Fire, 0.0 = Industrial Fire
        forest_fire_dir = self.root_dir / 'forest_fire'
        if forest_fire_dir.exists():
            for img_name in os.listdir(forest_fire_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(forest_fire_dir / img_name)
                    self.labels.append(1.0)
                    
        industrial_fire_dir = self.root_dir / 'industrial_fire'
        if industrial_fire_dir.exists():
            for img_name in os.listdir(industrial_fire_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(industrial_fire_dir / img_name)
                    self.labels.append(0.0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(raw_data_dir, batch_size=32, val_split=0.2, num_workers=0):
    """
    Creates training and validation DataLoaders for the image dataset.
    """
    # Standard ResNet ImageNet transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # We create two instances of the dataset so we can apply different transforms
    full_dataset = FireImageDataset(raw_data_dir)
    
    if len(full_dataset) == 0:
        return None, None
        
    # Split dataset
    import torch
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Use random_split just for indices
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(FireImageDataset(raw_data_dir, transform=train_transform), train_indices)
    val_dataset = torch.utils.data.Subset(FireImageDataset(raw_data_dir, transform=val_transform), val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
