import numpy as np
import matplotlib.pyplot as plt
import deconv_utils as dcu


""" blank canvas, 256x256 """
image_size = (256, 256)
original_image = np.zeros(image_size)

square_size = 50

start_index = (image_size[0] - square_size) // 2
end_index = start_index + square_size

""" Drawing The Square """
original_image[start_index:end_index, start_index:end_index] = 1.0


#constants used throughout the demo
sigma = 8 #gaussian blur
noise_strength = 2.5 #added noise
K = 8 #for wiener filter
gamma = 0.1 #for CLS 
iterations = 100 #for RL 


"""
plt.figure()
plt.imshow(original_image, cmap="gray")
plt.title("Original Image")
plt.show()
"""

#Creating the Blur Kernel 

kernel_size = 15

""" Gaussian Blur Kernel """

#creating the number line for the gaussian kernel
kernel_center = kernel_size // 2
x = np.linspace(-kernel_center, kernel_center, kernel_size)
y = np.linspace(-kernel_center, kernel_center, kernel_size)

xx, yy = np.meshgrid(x, y) #for radially symmetric kernels

##Gaussian Function
gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

""" Creating Motion Blur Kernel """
motion_kernel = np.zeros((kernel_size, kernel_size))
center_row = kernel_size // 2
motion_kernel[center_row, : ] = 1.0
motion_kernel = motion_kernel / np.sum(motion_kernel) #normalization


kernel = gaussian_kernel #set the kernel

"""
plt.figure()
plt.imshow(kernel, cmap='gray')
plt.title(f"{kernel_size}x{kernel_size} Gaussian Kernel")
plt.show()
"""

#padding 

padded_kernel = np.zeros(image_size) #create a canvas identical to the original image size
start_pad = (image_size[0] - kernel_size) // 2
end_pad = start_pad + kernel_size
padded_kernel[start_pad:end_pad, start_pad:end_pad] = kernel

#rolling the kernel to move the center to (0, 0)
rolled_padded_kernel = np.fft.ifftshift(padded_kernel)

#get H in the freq. domain (fft)
H = np.fft.fft2(rolled_padded_kernel)

#Convolving in Frequency Domain (Multiplication)
F_image = np.fft.fft2(original_image)

F_blur = F_image * H

#invert fft to get the blurry image back
blurry_image = np.real(np.fft.fft2(F_blur))

"""
plt.figure()
plt.imshow(blurry_image, cmap='gray')
plt.title("Blurred Image (Convolution)")
plt.show()
"""


"""     Brute Force Deconvolution (FAIL)      """
#simulate a noise on the images

noise = np.random.normal(0, noise_strength, size=image_size)
noisy_blurry_image = blurry_image + noise

"""
#plot the new noisy image
plt.figure()
plt.imshow(noisy_blurry_image, cmap='gray')
plt.title("Noisy Blurry Image")
plt.show()
"""

#Brute Force Deconvolution
F_blur_noisy = np.fft.fft2(noisy_blurry_image)

#divide by H to "deconvolve"
#add a tiny number a to avoid division by zero
a = 1e-9 
F_deconv_brute = F_blur_noisy / (H + a)

#invert the image
deconv_image_brute = np.real(np.fft.fft2(F_deconv_brute))

"""
#BEHOLD THE FAILURE
plt.figure()
plt.imshow(deconv_image_brute, cmap='gray')
plt.title("Brute-Force Deconvolution")
plt.show()
"""



"""     Noise Aware Wiener Filter Demo     """

#calculating wiener filter components H* and |H|**2
H_conj = np.conjugate(H)
H_mag_sq = np.abs(H)**2


#wiener function W
W = H_conj / (H_mag_sq + K)

#applying the wiener filter 
F_deconv_wiener = F_blur_noisy * W

#invert fft to get the image 
deconv_image_wiener = np.real(np.fft.ifft2(F_deconv_wiener))


""" Constrained Least Squares Demo """
#a 3x3 laplacian kernel
p = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
])
#padding the laplacian kernel
padded_laplacian = np.zeros(image_size)
start_lap = (image_size[0] - 3) // 2
end_lap = start_lap + 3
padded_laplacian[start_lap:end_lap, start_lap:end_lap] = p
rolled_padded_laplacian = np.fft.ifftshift(padded_laplacian)#roll to the origin
P = np.fft.fft2(rolled_padded_laplacian)

#building the cls function

#we have H_conj and H_mag_sq from weiner 
P_mag_sq = np.abs(P)**2

CLS = H_conj / (H_mag_sq + gamma * P_mag_sq)

##Applying the filter
F_deconv_cls = F_blur_noisy * CLS
deconv_image_cls = np.real(np.fft.ifft2(F_deconv_cls))

RL = dcu.deconv_rl(noisy_blurry_image, H, iterations)


"""
plt.figure()
plt.imshow(deconv_image_wiener, cmap='gray', vmin=0, vmax=1) # vmin/vmax cleans up display
plt.title(f"Wiener Deconvolution (K={K})")
plt.show()
"""

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

ax = axes[0 ,0]
ax.imshow(original_image, cmap="gray")
ax.set_title("Original Image")

ax = axes[0 ,1]
ax.imshow(kernel, cmap="gray")
ax.set_title(f"Blur Kernel, sigma = {sigma}")

ax = axes[0, 2]
ax.imshow(blurry_image, cmap="gray")
ax.set_title("Convolved with Gaussian Blur Kernel")

ax = axes[1, 0]
ax.imshow(noisy_blurry_image, cmap="gray")
ax.set_title(f"Blurry Image with Noise (n = {noise_strength})")

ax = axes[1, 1]
ax.imshow(deconv_image_brute, cmap="gray")
ax.set_title("Brute Deconvolution (with noise)") 

ax = axes[1, 2]
ax.imshow(deconv_image_wiener, cmap="gray", vmin = 0, vmax = 1)
ax.set_title("Deconvolution with Wiener filter (High K Values)")

ax = axes[2, 0]
ax.imshow(deconv_image_cls, cmap="gray", vmin = 0, vmax = 1)
ax.set_title(f"Deconvolution with CLS (gamma = {gamma})")

ax = axes[2, 1]
ax.imshow(RL, cmap = "gray", vmin = 0, vmax = 1)
ax.set_title(f"Deconvolution Richardson-Lucy(iter = {iterations})")

plt.subplots_adjust(hspace=0.4)
plt.show()