"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    """Create training and testing DataLoader
    
    Takes in a training and testing directory path and turns them into PyTorch Datasets and then into PyTorch DataLoaders
    
    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data
        batch_size: Number of sample per batch in each of the dataloaders
        num_workers: An integer fro number of workers per DataLoader.
    
    Returns:
        A tupple of (train_dataloader, test_dataloader, class_names)
        Example usage:
        train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
                                                                            test_dir=path/to/test_dir,
                                                                            transform=some_transform,
                                                                            batch_size=32,
                                                                            num_workers=4)
    """

    # use ImageFolder to create datasets
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn image into dataloaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, test_dataloader, class_names