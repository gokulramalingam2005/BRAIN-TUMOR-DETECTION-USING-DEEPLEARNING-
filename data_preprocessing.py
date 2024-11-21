import os
import cv2
import numpy as np

def preprocess_image(image_path, target_size):
    """
    Preprocesses an MRI image for model input.

    Args:
    - image_path: Path to the MRI image file.
    - target_size: Tuple specifying the target size (height, width) for resizing the image.

    Returns:
    - Preprocessed image as a NumPy array.
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize the image to the target size
    image = cv2.resize(image, target_size)
    
    # Normalize the pixel values to the range [0, 1]
    image = image / 255.0
    
    return image

def load_dataset(data_dir, target_size):
    """
    Loads the dataset of MRI images and their corresponding labels.

    Args:
    - data_dir: Path to the dataset directory.
    - target_size: Tuple specifying the target size (height, width) for resizing the images.

    Returns:
    - List of preprocessed images as NumPy arrays.
    - List of corresponding labels (0 for 'NO' class, 1 for 'YES' class).
    """
    images = []
    labels = []

    # Iterate through the subdirectories (classes) in the dataset directory
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        # Get the label for the current class
        label = 0 if class_name == 'NO' else 1
        
        # Iterate through the images in the current class directory
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            
            # Preprocess the image
            image = preprocess_image(image_path, target_size)
            
            # Append the preprocessed image and its label to the lists
            images.append(image)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Example usage
if __name__ == "__main__":
    # Define the directory containing the dataset
    dataset_dir = 'path/to/dataset'
    
    # Define the target size for resizing the images
    target_size = (128, 128)
    
    # Load the dataset
    images, labels = load_dataset(dataset_dir, target_size)
    
    # Print the shape of the loaded data
    print("Loaded images shape:", images.shape)
    print("Loaded labels shape:", labels.shape)
