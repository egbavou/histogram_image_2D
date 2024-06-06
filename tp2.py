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
    
def adjust_dynamic_range(image, a=None, b=None):
    
    image_data = list(image.getdata())
    
    if a is None:
        a = min(image_data)
    if b is None:
        b = max(image_data)
    
    adjusted_data = [
        int((pixel - a) / (b - a) * 255) if a <= pixel <= b else (0 if pixel < a else 255)
        for pixel in image_data
    ]
    
    adjusted_image = Image.new('L', image.size)
    adjusted_image.putdata(adjusted_data)
    
    return adjusted_image
    
def highlight_range(image, a, b, f):
    
    # Convert image to list of pixels
    image_data = list(image.getdata())
    
    # Highlight the range
    highlighted_data = [
        255 if a <= pixel <= b else (pixel if f == 1 else 0)
        for pixel in image_data
    ]
    
    highlighted_image = Image.new('L', image.size)
    highlighted_image.putdata(highlighted_data)
    
    return highlighted_image  

def dilate_contract(image, A, ad, ac):
    
    if ad + ac != 1:
        raise ValueError("The sum of ad and ac must be 1")
    
    image_data = list(image.getdata())
    
    processed_data = [
        int(A + (pixel - A) * ad) if pixel > A else int(A - (A - pixel) * ac)
        for pixel in image_data
    ]
    
    processed_image = Image.new('L', image.size)
    processed_image.putdata(processed_data)
    
    return processed_image

def equalize_histogram(image):
    
    image_data = list(image.getdata())
    
    # Calculate the histogram
    histogram = [0] * 256
    for pixel in image_data:
        histogram[pixel] += 1
    
    # Calculate the cumulative distribution function (CDF)
    cdf = [0] * 256
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]
    
    # Normalize the CDF
    cdf_min = min(cdf)
    cdf_max = max(cdf)
    cdf_normalized = [(cdf[i] - cdf_min) / (cdf_max - cdf_min) * 255 for i in range(256)]
    
    # Map the original image data using the normalized CDF
    equalized_data = [int(cdf_normalized[pixel]) for pixel in image_data]
    
    # Create the equalized image
    equalized_image = Image.new('L', image.size)
    equalized_image.putdata(equalized_data)
    
    return equalized_image

def calculate_histogram(image):
    
    image_data = list(image.getdata())
    histogram = [0] * 256
    for pixel in image_data:
        histogram[pixel] += 1
    
    return histogram

if __name__ == "__main__":
    
    image_paths = ['rice.tif','spine.tif']
    
    NIV = 256 
    
    for image_path in image_paths:
        
        
        image = Image.open(image_path).convert('L')
        
        if image.mode != 'L':
            raise ValueError("The image must be in grayscale")

        visualize_histograms(image, NIV)
        
        adjusted_image = adjust_dynamic_range(image)
        adjusted_image.show()
        adjusted_image.save(f"adjusted_image_{image_path.replace('.tif','')}.jpg")
        
        highlighted_image = highlight_range(image, 100, 150, 1)
        highlighted_image.show()
        highlighted_image.save(f"highlighted_image_{image_path.replace('.tif','')}.jpg")
        
        processed_image = dilate_contract(image, A=128, ad=0.7, ac=0.3)
        processed_image.show()
        processed_image.save(f"processed_image_{image_path.replace('.tif','')}.jpg")
        
        
        equalized_image = equalize_histogram(image)
        
        original_histogram = calculate_histogram(image)
        equalized_histogram = calculate_histogram(equalized_image)
        
        image.show()
        equalized_image.show()
        
        equalized_image.save(f"equalized_image_{image_path.replace('.tif','')}.jpg")
        
        print("Original Histogram:", original_histogram)
        print("Equalized Histogram:", equalized_histogram)
   
