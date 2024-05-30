from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def compute_histograms(image, NIV):
    
    if len(image.shape) != 2:
        raise ValueError("The image must be in grayscale")
    
    histogram = np.zeros(NIV, dtype=int)
    for pixel_value in image.flatten():
        histogram[pixel_value] += 1
    
    cumulative_histogram = [0] * len(histogram)
    cumulative_sum = 0
    for i in range(len(histogram)):
        cumulative_sum += histogram[i]
        cumulative_histogram[i] = cumulative_sum
    
    return histogram, cumulative_histogram

def visualize_histograms(image_path, NIV):
    
    image = Image.open(image_path).convert('L')
    # image.show()
    image_array = np.array(image)
    
    histogram, cumulative_histogram = compute_histograms(image_array, NIV)
    
    plt.figure(figsize=(12, 5))
        
    plt.subplot(1, 2, 1)
    plt.bar(range(NIV), histogram)
    plt.xlabel('Gray level')
    plt.ylabel('Frequency')
    plt.title(f'Histogram for {image_path}')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(NIV), cumulative_histogram, color='r')
    plt.xlabel('Gray level')
    plt.ylabel('Cumulative frequency')
    plt.title(f'Cumulative Histogram for {image_path}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    image_paths = ['cameraman.tif','lena.jpg']

    NIV = 256 
    
    for image_path in image_paths:

        visualize_histograms(image_path, NIV)
