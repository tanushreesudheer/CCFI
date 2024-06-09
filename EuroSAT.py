import os
from torch.utils.data import Dataset
from PIL import Image

class EuroSAT_data(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Initialize EuroSAT dataset.

        Args:
            dataset_path (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Get the list of classes in the dataset directory
        self.classes = [class_name for class_name in os.listdir(dataset_path) if not class_name.startswith('.')]  

        # Count the number of images for each class
        self.class_image_counts = {}
        for class_name in self.classes:
            class_dir = os.path.join(dataset_path, class_name)
            self.class_image_counts[class_name] = len([file for file in os.listdir(class_dir) if file.endswith('.jpg')])  
        
        # Calculate the total number of images in the dataset
        self.total_images = sum(self.class_image_counts.values())

    def __getitem__(self, index):
        """
        Get the item at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image and its class index.
        """
        total_images = self.total_images
        class_index = 0
        
        # Determine the class index based on the index of the item
        for class_name, count in self.class_image_counts.items():
            if index < count:
                break
            index -= count
            class_index += 1
    
        class_name = self.classes[class_index]

        # Adjust index to start from 1 if it's greater than or equal to 0
        if index >= 0:
            index += 1
    
        # Construct the path to the image file
        image_path = os.path.join(self.dataset_path, class_name, f"{class_name}_{index}.jpg")
        image = Image.open(image_path).convert('RGB')

        # Apply transformation to the image if specified
        if self.transform:
            image = self.transform(image)

        return image, class_index


    def __len__(self):
        """
        Get the length of the accessed class.

        Returns:
            int: Total number of images in the dataset.
        """
        return self.total_images
