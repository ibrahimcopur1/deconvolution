import numpy as np  # noqa: F401
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt 
image = np.zeros((256, 256), dtype = np.float64) 

center_x, center_y = 128, 128
size = 50
start_x = center_x - size // 2
start_y = center_y - size // 2
end_x = start_x + size
end_y = start_y + size

image[start_y:end_y, start_x:end_x] = 255.0

""" Gaussian Kernel """

def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center) 

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel = kernel / np.sum(kernel)
    return kernel

blur_kernel = create_gaussian_kernel(size = 15, sigma = 3.0)

def convolve_fft(image: np.ndarray, kernel:np.ndarray) -> np.ndarray:
    im_h, im_w = image.shape
    k_h, k_w = kernel.shape
    pad_h = k_h - 1
    pad_w = k_w - 1

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

    padded_kernel = np.zeros_like(padded_image, dtype=np.float64)
    kernel_shifted = fftshift(kernel)

    padded_kernel[:k_h, :k_w] = kernel_shifted[:k_h, :k_w]

    fft_image = fft2(padded_image)
    fft_kernel = fft2(padded_kernel)

    fft_convolved = fft_image * fft_kernel

    convolved_image_padded = np.real(ifft2(fft_convolved))

    unpadded_image = convolved_image_padded[pad_top:pad_top + im_h, pad_left:pad_left + im_w]

    return unpadded_image


blurry_image = convolve_fft(image, blur_kernel)



plt.subplot(1, 3, 1) # First subplot
plt.imshow(image, cmap='gray') # Use a grayscale colormap
plt.title("Original Image")
plt.axis('off') # Hide axes ticks

plt.subplot(1, 3, 2) # Second subplot
plt.imshow(blur_kernel, cmap='gray')
plt.title("Gaussian Blur Kernel")
plt.axis('off')

plt.subplot(1, 3, 3) # Third subplot
plt.imshow(blurry_image, cmap='gray')
plt.title("Blurry Image")
plt.axis('off')

plt.show()