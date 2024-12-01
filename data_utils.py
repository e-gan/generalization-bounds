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

        # Corruption transformation initialization
        if corruption_type == "random_labels":
            self.corrupt_labels(1)
        elif corruption_type == "partially_corrupted_labels":
            self.corrupt_labels(self.corruption_prob)
        elif corruption_type == "gaussian_images":
            self.gaussian()
        elif corruption_type == "random_pixels":
            self.random_pixels()
        elif corruption_type == "shuffle_pixels":
            self.shuffle_pixels()


    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(10, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels

    def random_pixels(self):
        """ Shuffle the pixels of the image independently. """
        def randomize(image):
            perm = np.random.permutation(32 * 32 * 3)  # Unique permutation per image
            flat = image.flatten() # Flatten the image
            shuffled = flat[perm].reshape(32, 32, 3)  # Apply the permutation and reshape
            return shuffled
        new_data = np.array([randomize(image) for image in self.data])
        self.data = new_data

    def shuffle_pixels(self):
        """ Shuffle the pixels of the image using a fixed permutation. """
        perm = np.random.permutation(32 * 32 * 3)  # fixed permutation
        def shuffle(image):
            flat = image.flatten()  # Flatten the image
            shuffled = flat[perm].reshape(32, 32, 3)  # Apply the permutation and reshape
            return shuffled
        new_data = np.array([shuffle(image) for image in self.data])
        self.data = new_data


    def gaussian(self):
        """ Replace the image with Gaussian noise having the same mean and variance. """
        def add_gaussian(image):
            mean = image.mean(axis=(0, 1))  # Compute per-channel mean
            std = image.std(axis=(0, 1))   # Compute per-channel standard deviation
            gaussian_data = np.random.normal(loc=mean, scale=std, size=image.shape)

            # Ensure data is within the valid range [0, 255] and type uint8
            gaussian_data = np.clip(gaussian_data, 0, 255).astype(np.uint8)
            return gaussian_data

        # Generate Gaussian noise for all images in the dataset
        new_data = np.array([add_gaussian(image) for image in self.data])

        # Update the dataset
        self.data = new_data

    
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

def visualize_dataset(dataset, num_samples=10, original_dataset=None, title=""):
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
            subtitle = f"Corrupted: {label}\nOriginal: {original_label}"
        else:
            subtitle = f"Label: {label}"
        
        # Plot the image
        axes[i].imshow(image)  # Image is now a PIL image
        axes[i].axis('off')
        axes[i].set_title(subtitle, fontsize=8)
    
    plt.tight_layout()
    fig.suptitle(title)
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
    datasets = [original_dataset, random_dataset, partial_dataset, shuffled_dataset, random_pixel_dataset, gaussian_dataset, test_dataset]

    train_loader = get_train_dataloader(corruption_type='random_labels')
    test_loader = get_test_dataloader()
    print("created dataloaders")
    print("generating visualizations")
    visualize_dataset(dataset=random_dataset, original_dataset=original_dataset, title="Random Labels")
    visualize_dataset(dataset=random_pixel_dataset, original_dataset=original_dataset, title="Random Pixels")
    visualize_dataset(dataset=partial_dataset, original_dataset=original_dataset, title="Corrupted Labels by 0.6")
    visualize_dataset(dataset=gaussian_dataset, original_dataset=original_dataset, title="Gaussian Inputs")
    visualize_dataset(dataset=shuffled_dataset, original_dataset=original_dataset, title="Shuffled Pixels")

    print("checking normalization")
    for d in datasets:
        check_normalization(d)