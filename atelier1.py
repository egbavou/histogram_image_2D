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

def visualize_histograms(image, NIV):
    
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


def crop_image(image, left, top, right, bottom):
    
    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        raise ValueError("Cropping coordinates are out of image bounds")
    
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def binarize_image(image, threshold):

    if image.mode != 'L':
        raise ValueError("The image must be in grayscale")
    
    binarized_image = image.point(lambda p: 255 if p > threshold else 0)
    return binarized_image

if __name__ == "__main__":
    
    image_paths = ['rice.tif','spine.tif']

    NIV = 256 
    
    for image_path in image_paths:
        
        image = Image.open(image_path).convert('L')
        
        # 1°)
        visualize_histograms(image, NIV)

        # 2°)
        left = 50
        top = 50
        right = 200
        bottom = 200
        
        cropped_image = crop_image(image, left, top, right, bottom)
        
        cropped_image.show()
        
        cropped_image.save(f"cropped_{image_path.replace('.tif','')}.jpg")
        
        # 3°)
        threshold = 128
        
        binarized_image = binarize_image(image, threshold)
        
        binarized_image.show()
        
        binarized_image.save(f"binarized_{image_path.replace('.tif','')}.jpg")

        
        
