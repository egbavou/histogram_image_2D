import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def display_fft(image_path):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
   
    fft_img = np.fft.fftshift(np.fft.fft2(img_np))
    
    magnitude_spectrum = 20 * np.log(np.abs(fft_img))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f'Image originale {image_path}')
    plt.imshow(img_np, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Spectre de magnitude FFT {image_path}')
    plt.imshow(magnitude_spectrum, cmap='gray')
    
    plt.show()

display_fft('mit.tif')
display_fft('cameraman.tif')

def reconstruct_with_zero_phase(image_path):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    
    fft_img = np.fft.fftshift(np.fft.fft2(img_np))
   
    magnitude = np.abs(fft_img)
    zero_phase_img = np.fft.ifft2(np.fft.ifftshift(magnitude)).real
   
    plt.figure(figsize=(6, 6))
    plt.title(f'Image reconstituée (phase annulée) {image_path}')
    plt.imshow(zero_phase_img, cmap='gray')
    plt.show()

reconstruct_with_zero_phase('mit.tif')
reconstruct_with_zero_phase('cameraman.tif')

def reconstruct_with_constant_magnitude(image_path):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    
    fft_img = np.fft.fftshift(np.fft.fft2(img_np))

    phase = np.angle(fft_img)
    mean_magnitude = np.mean(np.abs(fft_img))
    constant_magnitude_img = np.fft.ifft2(np.fft.ifftshift(mean_magnitude * np.exp(1j * phase))).real
    
    plt.figure(figsize=(6, 6))
    plt.title(f'Image reconstituée (module constant) {image_path}')
    plt.imshow(constant_magnitude_img, cmap='gray')
    plt.show()

reconstruct_with_constant_magnitude('mit.tif')
reconstruct_with_constant_magnitude('cameraman.tif')

def resize_image_to_match(img1, img2):
    img2_resized = img2.resize(img1.size, Image.BILINEAR)
    return img2_resized

def create_image_from_phase_and_magnitude(phase_image_path, magnitude_image_path):
    phase_img = Image.open(phase_image_path).convert('L')
    magnitude_img = Image.open(magnitude_image_path).convert('L')
    
    magnitude_img = resize_image_to_match(phase_img, magnitude_img)
    
    phase_np = np.array(phase_img)
    magnitude_np = np.array(magnitude_img)
    
    fft_phase = np.fft.fftshift(np.fft.fft2(phase_np))
    fft_magnitude = np.fft.fftshift(np.fft.fft2(magnitude_np))
    
    combined_fft = np.abs(fft_magnitude) * np.exp(1j * np.angle(fft_phase))
    combined_img = np.fft.ifft2(np.fft.ifftshift(combined_fft)).real
    
    plt.figure(figsize=(6, 6))
    plt.title('Image créée à partir du module de cameraman et de la phase de mit')
    plt.imshow(combined_img, cmap='gray')
    plt.show()

create_image_from_phase_and_magnitude('mit.tif', 'cameraman.tif')