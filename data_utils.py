import torch
from torchvision import datasets, transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, RandomSampler
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class PerImageWhitening:
    def __call__(self, image):
        # Ensure the tensor is contiguous
        image = image.contiguous().view(3, -1)
        
        # Calculate per-image mean and adjusted standard deviation
        mean = image.mean(dim=1, keepdim=True)
        std = image.std(dim=1, keepdim=True)
        adjusted_std = torch.maximum(std, torch.tensor(1.0 / (28 * 28), device=std.device))
        
        # Normalize image
        whitened = (image - mean) / adjusted_std
        return whitened.view(3, 28, 28)
    
    
class CorruptedCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, download=True, corruption_type=None, corruption_prob=0):
        """
        Custom CIFAR10 class that applies a specified transformation (random labels, partially corrupted, etc.)
        to the dataset before applying other transformations like ToTensor, PerImageWhitening, CenterCrop.
        """
        self.image_transforms = transforms.Compose([
            transforms.CenterCrop(28),            # Center crop to 28x28
            transforms.ToTensor(),     # Convert PIL image to Tensor and scale to [0, 1]
            PerImageWhitening()        # Apply per-image whitening
        ])
        super().__init__(root=root, transform=self.image_transforms, train=train, download=download)

        # Set corruption transformation based on input type
        self.corruption_type = corruption_type
        self.corruption_prob = corruption_prob
        self.perm = None

        # Corruption transformation initialization
        if corruption_type == "random_labels":
            self.corruption_transform = self.random_labels
        elif corruption_type == "partially_corrupted_labels":
            self.corruption_transform = self.partially_corrupted_labels
        elif corruption_type == "gaussian_images":
            self.corruption_transform = self.gaussian_images
        elif corruption_type == "random_pixels":
            self.corruption_transform = self.random_pixels
        elif corruption_type == "shuffle_pixels":
            self.corruption_transform = self.shuffle_pixels
        else:
            self.corruption_transform = None


    def random_labels(self, image, label):
      """ Replace the label with a random class label. """
      new_label = torch.randint(0, 10, (1,)).item()  # Random class label from 0 to 9
      return image, new_label

    def shuffle_pixels(self, image, label):
        """ Shuffle the pixels of the image using a fixed permutation. """
        if self.perm is None:
            self.perm = np.random.permutation(28 * 28 * 3)  # Generate a fixed permutation for all images
        flat = image.reshape(-1)  # Flatten the image
        shuffled = flat[self.perm].reshape(3, 28, 28)  # Apply the permutation and reshape
        return torch.tensor(shuffled, dtype=torch.float32), label  # Convert to tensor

    def random_pixels(self, image, label):
        """ Shuffle the pixels of the image independently. """
        perm = np.random.permutation(28 * 28 * 3)  # Unique permutation per image
        flat = image.reshape(-1)  # Flatten the image
        shuffled = flat[perm].reshape(3, 28, 28)  # Apply the permutation and reshape
        return torch.tensor(shuffled, dtype=torch.float32), label  # Convert to tensor

    def gaussian_images(self, image, label):
        """ Replace the image with Gaussian noise having the same mean and variance. """
        mean = image.mean(axis=(0, 1))
        std = image.std(axis=(0, 1))
        gaussian_data = np.random.normal(loc=mean, scale=std, size=image.shape)
        return torch.tensor(gaussian_data, dtype=torch.float32), label  # Convert to tensor

    def partially_corrupted_labels(self, image, label):
        """ Corrupt the label with a probability `corruption_prob`. """
        if np.random.rand() < self.corruption_prob:
            new_label = np.random.randint(0, 10)
            while new_label == label:
                new_label = np.random.randint(0, 10)
            return image, new_label
        return image, label

    def __getitem__(self, index):
        """
        Override __getitem__ to apply all the transformations (crop, corruption, tensor conversion, whitening).
        """
        image, label = super().__getitem__(index)
        if self.train and self.corruption_transform:
            image, label = self.corruption_transform(image, label)

        return image, label
    
def examine_dataset(dataset):
    """
    Examine a dataset to compute and display statistics such as mean, std, and visualize some sample images.
    
    Args:
        dataset: The dataset to examine (e.g., CIFAR-10 with transformations applied).
    """
    # Convert dataset into a tensor to compute statistics
    all_images = []
    for idx in range(len(dataset)):
        image, _ = dataset[idx]
        all_images.append(image)
    
    all_images = torch.stack(all_images)
    
    # Compute the mean and std across the dataset
    mean = all_images.mean(dim=(0, 2, 3))
    std = all_images.std(dim=(0, 2, 3))
    
    # Print out the results
    print(f"Dataset Statistics:")
    print(f"Mean per channel: {mean}")
    print(f"Standard deviation per channel: {std}")
    
    # Visualize some images from the dataset to confirm transformations
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        image, label = dataset[i]
        ax.imshow(image.permute(1, 2, 0).numpy())  # Convert from CxHxW to HxWxC
        ax.axis('off')
        ax.set_title(f"Label: {label}")
    plt.show()

def check_normalization(dataset):
    """
    Check if the images in the dataset are normalized.
    
    Args:
        dataset: The dataset to check.
    """
    # Stack all images in the dataset to compute mean and std
    all_images = []
    for idx in range(len(dataset)):
        image, _ = dataset[idx]
        all_images.append(image)
    
    all_images = torch.stack(all_images)
    
    # Compute mean and std across the dataset
    mean = all_images.mean(dim=(0, 2, 3))  # Mean across (batch, height, width)
    std = all_images.std(dim=(0, 2, 3))    # Std across (batch, height, width)
    
    # Print out the results
    print(f"Mean per channel (should be close to 0 if normalized): {mean}")
    print(f"Standard deviation per channel (should be close to 1 if normalized): {std}")

def visualize_dataset(dataset, num_samples=10, original_dataset=None):
    """
    Visualize images and labels from a given dataset.

    Args:
        dataset: The dataset to visualize (e.g., corrupted dataset).
        num_samples: Number of samples to visualize.
        original_dataset: The original dataset for comparison (optional).
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Reverse the normalization for visualization (scale back to [0, 255])
        image = image * 255  # Scale back to [0, 255]
        image = torch.clamp(image, 0, 255)  # Ensure the values are within [0, 255]
        
        # Convert from PyTorch tensor to NumPy array (C, H, W) -> (H, W, C)
        image = image.permute(1, 2, 0).byte().numpy()  # Ensure correct data type (byte) for image
        
        # Convert to PIL Image (Ensure values are in uint8 and range [0, 255])
        image = Image.fromarray(image.astype(np.uint8))  # Convert to uint8 before creating PIL image
        
        # Compare to the original label if provided
        if original_dataset:
            original_label = original_dataset.targets[idx]
            title = f"Corrupted: {label}\nOriginal: {original_label}"
        else:
            title = f"Label: {label}"
        
        # Plot the image
        axes[i].imshow(image)  # Image is now a PIL image
        axes[i].axis('off')
        axes[i].set_title(title, fontsize=8)
    
    plt.tight_layout()
    plt.show()



def get_train_dataloader(dataset="CIFAR10",
                         batch_size=16,
                         num_workers=2,
                         loss_fn='MSE',
                         shuffle=True,
                         bound_num_batches=None,
                         corruption_type=None,
                         corruption_prob=0.6,
                         num_classes=10):
    """ return train dataloader
    """
    if dataset=="CIFAR10":
        train_dataset = CorruptedCIFAR10(root="./data", train=True, download=True, corruption_type=corruption_type, corruption_prob=corruption_prob)

    #need to onehot encode labels for MSE loss    
    if loss_fn == 'MSE':
        train_dataset.targets = torch.nn.functional.one_hot(torch.tensor(train_dataset.targets), num_classes=num_classes).float()

    if bound_num_batches is not None:  # sampling for the bound
        sampler = RandomSampler(train_dataset,
                                replacement=True,
                                num_samples=bound_num_batches)
        training_loader = DataLoader(train_dataset,
                                     sampler=sampler,
                                     num_workers=num_workers,
                                     batch_size=batch_size)
    else:
        training_loader = DataLoader(train_dataset,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     batch_size=batch_size)

    return training_loader


def get_test_dataloader(dataset="CIFAR10",
                        batch_size=16,
                        num_workers=2,
                        loss_fn='MSE',
                        shuffle=False,
                        num_classes=10):
    """ return test dataloader
    """
    if dataset=="CIFAR10":
        test_dataset = CorruptedCIFAR10(root="./data", train=False, download=True)

    if loss_fn == 'MSE':
        test_dataset.targets = torch.nn.functional.one_hot(torch.tensor(test_dataset.targets), num_classes=num_classes).float()

    test_loader = DataLoader(test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader

if __name__ == "__main__":
    original_dataset = CorruptedCIFAR10(root="./data", train=True, download=True, corruption_type=None)
    random_dataset = CorruptedCIFAR10(root="./data", train=True, download=True, corruption_type="random_labels")
    partial_dataset = CorruptedCIFAR10(root="./data", train=True, download=True, corruption_type="partially_corrupted_labels", corruption_prob=0.6)
    shuffled_dataset = CorruptedCIFAR10(root="./data", train=True, download=True, corruption_type="shuffle_pixels")
    random_pixel_dataset = CorruptedCIFAR10(root="./data", train=True, download=True, corruption_type="random_pixels")
    gaussian_dataset = CorruptedCIFAR10(root="./data", train=True, download=True, corruption_type="gaussian_images")
    test_dataset = CorruptedCIFAR10(root="./data", train=False, download=True)
    print("created datasets")

    train_loader = get_train_dataloader(corruption_type='random_labels')
    test_loader = get_test_dataloader()
    print("created dataloaders")
    print("generating visualizations")
    visualize_dataset(dataset=random_dataset, original_dataset=original_dataset)