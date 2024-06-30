from PIL import Image, ImageOps  # Importing necessary libraries for image processing
import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for plotting histograms
from scipy.ndimage import label, find_objects  # Importing functions for labeling and finding objects in the image
from scipy.ndimage import generate_binary_structure  # Importing function to generate a binary structure
import pandas as pd
from scipy.ndimage import center_of_mass
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from cv2 import dilate


def show_image(image):
    """
    Displays a given image using matplotlib.
    
    Parameters:
    - image: numpy array or PIL image to be displayed.
    """
    plt.imshow(image, cmap='gray')  # Display the image in grayscale
    plt.axis('off')  # Hide the axis
    plt.show()  # Show the plot

def compute_histograms(image, NIV):
    """
    Computes the histogram and cumulative histogram of a grayscale image.
    
    Parameters:
    - image: 2D numpy array representing a grayscale image.
    - NIV: Number of intensity values (usually 255 for 8-bit images).
    
    Returns:
    - histogram: Array containing the frequency of each intensity value.
    - cumulative_histogram: Array containing the cumulative frequency of intensity values.
    """
    if len(image.shape) != 2:  # Check if the image is grayscale
        raise ValueError("The image must be in grayscale")
    
    histogram = np.zeros(NIV, dtype=int)  # Initialize histogram with zeros
    for pixel_value in image.flatten():  # Iterate over each pixel value in the image
        histogram[pixel_value] += 1  # Increment the corresponding histogram bin
    
    cumulative_histogram = [0] * len(histogram)  # Initialize cumulative histogram
    cumulative_sum = 0  # Initialize cumulative sum
    for i in range(len(histogram)):  # Compute the cumulative histogram
        cumulative_sum += histogram[i]
        cumulative_histogram[i] = cumulative_sum
    
    return histogram, cumulative_histogram  # Return the histogram and cumulative histogram

def visualize_histograms(image, image_path, NIV):
    """
    Visualizes the histogram and cumulative histogram of a given grayscale image.
    
    Parameters:
    - image: PIL image to be analyzed.
    - image_path: Path or identifier for the image (used in plot titles).
    - NIV: Number of intensity values (usually 255 for 8-bit images).
    """
    image_array = np.array(image)  # Convert the image to a numpy array
    
    histogram, cumulative_histogram = compute_histograms(image_array, NIV)  # Compute the histograms
    
    plt.figure(figsize=(12, 5))  # Create a figure with a specific size
        
    plt.subplot(1, 2, 1)  # Create a subplot for the histogram
    plt.bar(range(NIV), histogram)  # Plot the histogram
    plt.xlabel('Gray level')  # Set x-axis label
    plt.ylabel('Frequency')  # Set y-axis label
    plt.title(f'Histogram for {image_path}')  # Set the title
    
    plt.subplot(1, 2, 2)  # Create a subplot for the cumulative histogram
    plt.bar(range(NIV), cumulative_histogram, color='r')  # Plot the cumulative histogram
    plt.xlabel('Gray level')  # Set x-axis label
    plt.ylabel('Cumulative frequency')  # Set y-axis label
    plt.title(f'Cumulative Histogram for {image_path}')  # Set the title
    
    plt.tight_layout()  # Adjust the layout
    plt.show()  # Show the plot

def combined_binary_np(image, threshold1=127, threshold2=200):
    """
    Generates a combined binary image based on two threshold values.
    
    Parameters:
    - image: PIL image to be binarized.
    - threshold1: First threshold value.
    - threshold2: Second threshold value.
    
    Returns:
    - combined_binary: Numpy array representing the combined binary image.
    """
    binary1 = image.point(lambda p: p > threshold1 and 255)  # Apply first threshold
    binary2 = image.point(lambda p: p > threshold2 and 255)  # Apply second threshold
    binary1_np = np.array(binary1)  # Convert the first binary image to numpy array
    binary2_np = np.array(binary2)  # Convert the second binary image to numpy array
    return np.logical_or(binary1_np, binary2_np)  # Combine the binary images using logical OR

def binarized(image, threshold1=127, threshold2=200):
    """
    Inverts a combined binary image to make it suitable for further processing.
    
    Parameters:
    - image: PIL image to be binarized.
    - threshold1: First threshold value.
    - threshold2: Second threshold value.
    
    Returns:
    - combined_binary: Inverted binary PIL image.
    """
    combined_binary_numpy = combined_binary_np(image, threshold1, threshold2) * 255  # Combine and scale the binary image
    combined_binary = Image.fromarray(combined_binary_numpy.astype(np.uint8))  # Convert the numpy array to PIL image
    return ImageOps.invert(combined_binary)  # Invert the binary image

def dilate_image(binary_image, kernel_size=(3, 3)):
    """
    Performs dilation on a binary image using a specified kernel size.

    Parameters:
    - binary_image: PIL.Image object
        The binary image to be dilated.
    - kernel_size: tuple of two ints, default (3, 3)
        The size of the kernel used for dilation.

    Returns:
    - dilated_image: PIL.Image object
        The dilated image.
    """
    # Convert the binary image from PIL format to a numpy array
    binary_image_np = np.array(binary_image)
    
    # Create a kernel (structuring element) for dilation
    kernel = np.ones(kernel_size, np.uint8)
    
    # Perform dilation on the numpy array image using the kernel
    # `iterations=1` indicates the dilation is applied once
    dilated_image_np = dilate(binary_image_np, kernel, iterations=1)
    
    # Convert the dilated numpy array back to a PIL image
    dilated_image = Image.fromarray(dilated_image_np)
    
    # Return the dilated PIL image
    return dilated_image

def isolate_digits(binary_image, save_path="digit_"):
    """
    Isolates individual digits from a binary image and saves them as separate images.
    
    Parameters:
    - binary_image: Binary PIL image containing digits.
    - save_path: Path prefix for saving isolated digit images.
    
    Returns:
    - isolated_digits: List of isolated digit images as PIL images.
    """
    binary_array = np.array(binary_image) // 255  # Convert the binary image to numpy array and scale to 0-1
    structure = generate_binary_structure(2, 1)  # Generate binary structure for labeling
    labeled_array, num_features = label(binary_array, structure=structure)  # Label connected components
    objects_slices = find_objects(labeled_array)  # Find the objects (digits)
    objects_slices = sorted(objects_slices, key=lambda x: x[1].start)  # Sort objects by their position
    isolated_digits = []  # Initialize list to store isolated digits
    for i, obj_slice in enumerate(objects_slices):  # Iterate over each object
        if obj_slice:
            isolated_digit = binary_array[obj_slice]  # Extract the isolated digit
            isolated_digit_image = Image.fromarray((isolated_digit * 255).astype(np.uint8))  # Convert to PIL image
            isolated_digit_image.save(f'{save_path}{i}.png')  # Save the isolated digit
            isolated_digits.append(isolated_digit_image)  # Append to the list
    return isolated_digits  # Return the list of isolated digits

def dilate(image_np, direction):
    """
    Performs dilation on an image in a specified direction.
    
    Parameters:
    - image_np: Numpy array representing the image to be dilated.
    - direction: Direction of dilation ('left', 'right', 'up', 'down').
    
    Returns:
    - dilated: Numpy array representing the dilated image.
    """
    height, width = image_np.shape  # Get the dimensions of the image
    dilated = np.zeros_like(image_np)  # Initialize an array for the dilated image

    if direction == 'left':  # Dilation to the left
        for x in range(height):
            for y in range(width):
                if image_np[x, y] == 255:
                    dilated[x, :y+1] = 255
    elif direction == 'right':  # Dilation to the right
        for x in range(height):
            for y in range(width-1, -1, -1):
                if image_np[x, y] == 255:
                    dilated[x, y:] = 255
    elif direction == 'up':  # Dilation upwards
        for y in range(width):
            for x in range(height):
                if image_np[x, y] == 255:
                    dilated[:x+1, y] = 255
    elif direction == 'down':  # Dilation downwards
        for y in range(width):
            for x in range(height-1, -1, -1):
                if image_np[x, y] == 255:
                    dilated[x:, y] = 255

    return dilated  # Return the dilated image

def find_cavities(image_np):
    """
    Identifies cavities in an image by performing dilations in all four directions.
    
    Parameters:
    - image_np: Numpy array representing the image to be analyzed.
    
    Returns:
    - cavities: Dictionary of numpy arrays representing cavities in different directions.
    """
    dilate_left = dilate(image_np, 'left')  # Dilate the image to the left
    dilate_right = dilate(image_np, 'right')  # Dilate the image to the right
    dilate_up = dilate(image_np, 'up')  # Dilate the image upwards
    dilate_down = dilate(image_np, 'down')  # Dilate the image downwards
    
    cavity_east = dilate_right & dilate_up & dilate_down & ~dilate_left  # Find cavities in the east
    cavity_west = dilate_left & dilate_up & dilate_down & ~dilate_right  # Find cavities in the west
    cavity_north = dilate_up & dilate_left & dilate_right & ~dilate_down  # Find cavities in the north
    cavity_south = dilate_down & dilate_left & dilate_right & ~dilate_up  # Find cavities in the south
    cavity_center = dilate_left & dilate_right & dilate_up & dilate_down  # Find cavities in the center

    cavities = {  # Create a dictionary of cavities
        'east': cavity_east & ~image_np,
        'west': cavity_west & ~image_np,
        'north': cavity_north & ~image_np,
        'south': cavity_south & ~image_np,
        'center': cavity_center & ~image_np
    }

    return cavities  # Return the dictionary of cavities

def save_cavities(cavities, i):
    """
    Saves the identified cavities as images and displays them.
    
    Parameters:
    - cavities: Dictionary of numpy arrays representing cavities in different directions.
    - i: Index of the digit (used for naming the saved images).
    """
    for direction, cavity in cavities.items():  # Iterate over each cavity direction
        cavity_image = Image.fromarray((cavity > 0).astype(np.uint8) * 255)  # Convert to PIL image
        cavity_image.save(f"cavityImages/cavity_digit_{i+1}_{direction}.png")  # Save the cavity image
        # print(f"Cavité du chiffre {i+1} - {direction}")  # Print the cavity information
        # show_image(cavity_image)  # Display the cavity image
        
        
if __name__ == "__main__":
    
    NIV = 255  # Number of intensity values for an 8-bit image
    image = Image.open("code0.tif").convert('L')  # Open and convert the image to grayscale
    image.show()  # Display the original image
    visualize_histograms(image, "code0.tif", NIV)  # Visualize the histograms
    
    binary_image = binarized(image)  # Binarize the image
    binary_image.save(f'project_binarized_image.png')
    isolated_digits = isolate_digits(binary_image)  # Isolate the digits
    
    for digit_index, digit_image in enumerate(isolated_digits):  # Process each isolated digit
        img = np.array(digit_image)  # Convert the digit image to numpy array
        cavities = find_cavities(img)  # Find the cavities
        save_cavities(cavities, digit_index)  # Save and display the cavities