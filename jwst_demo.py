import numpy as np
import deconv_utils as dcu
import image_io as io
import matplotlib.pyplot as plt 
from PIL import Image 

#input paths for images and psfs 
input_image = "images/hubble.jpg"
input_psf = "images/NIRCam_F070W_PSF.png"
psf_crop = 151

ISO = 500000
READ_NOISE = 0.001

USE_SYNTHETIC_STAR = False
USE_SYNTHETIC_PSF = False

if USE_SYNTHETIC_STAR:
    # Create a 500x500 black void with ONE bright pixel in the center
    print("DEBUG MODE: Using Synthetic Single-Pixel Star")
    original_image = np.zeros((500, 500))
    original_image[250, 250] = 100 # The perfect star
else:
    original_image = io.load_image(input_image)

    #star size and brightenss configuration 
    original_image[original_image < 0.1] = 0
    original_image = original_image ** 20
    original_image = original_image * 150

    
if USE_SYNTHETIC_PSF: 
    if original_image is not None:
        print(f"Loaded Image: {original_image.shape}")

        kernel = dcu.create_jwst_psf(psf_crop, 0.02)
        pad_w = psf_crop // 2
        padded_image = np.pad(original_image, pad_w, mode="constant", constant_values = 0)
        H = dcu.get_otf(kernel, padded_image.shape)

        F_photo = np.fft.fft2(padded_image)
        F_blur = F_photo * H
        padded_blurry = np.real(np.fft.ifft2(F_blur))
        blurry_photo = padded_blurry[pad_w:-pad_w, pad_w:-pad_w]

        noisy_blurry_photo = dcu.noise_poission_gaussian(blurry_photo, ISO, READ_NOISE)


else: 
    raw_psf = io.load_image(input_psf)

    if original_image is not None and raw_psf is not None: 
        print(f"Loaded Image: {original_image.shape}")
        print(f"Loaded Raw PSF: {raw_psf.shape}")

        #resizing and cropping and padding and whatnot
        psf_pil = Image.fromarray((raw_psf * 255).astype(np.uint8))
        psf_pil = psf_pil.resize((psf_crop, psf_crop), Image.LANCZOS)
        kernel = np.array(psf_pil).astype(float) / 255
        kernel = np.clip(kernel, 0, None)


        k_center = psf_crop // 2

        max_val = np.max(kernel)
        y_peaks, x_peaks = np.where(kernel == max_val)
        y_max = int(np.round(np.mean(y_peaks)))
        x_max = int(np.round(np.mean(x_peaks)))

        shift_y = k_center - y_max
        shift_x = k_center - x_max
        if shift_y != 0 or shift_x != 0:
            print(f"  -> Re-centering kernel (Shift: y={shift_y}, x={shift_x})...")
            kernel = np.roll(kernel, shift_y, axis=0)
            kernel = np.roll(kernel, shift_x, axis=1)
        
        kernel[kernel < 0.01] = 0

        kernel = kernel / np.sum(kernel) 

        pad_w = psf_crop // 2
        padded_photo = np.pad(original_image, pad_w, mode="reflect")
        padded_shape = padded_photo.shape

        H = dcu.get_otf(kernel, padded_shape)

        F_photo = np.fft.fft2(padded_photo)
        F_blur = F_photo * H
        padded_blurry = np.real(np.fft.ifft2(F_blur))

        blurry_photo = padded_blurry[pad_w: -pad_w, pad_w: -pad_w]

        noisy_blurry_photo = dcu.noise_poission_gaussian(blurry_photo, ISO, READ_NOISE)

        io.save_image_colored(noisy_blurry_photo, "output/hubble_F070W_conv_neg.png", "gray_r", False, 0, 1)


    
#DECONVOLUTION
noisy_padded = np.pad(noisy_blurry_photo, pad_w, mode="reflect")

deconv_padded = dcu.deconv_rl(noisy_padded, H, 150)

deconv_image = deconv_padded[pad_w:-pad_w, pad_w:-pad_w]

deconv_image[deconv_image < 0.01] = 0
deconv_image = deconv_image ** 10

#saving output images
#original modified image
io.save_image_colored(original_image, "output/hubble_original_neg.png", "gray_r", False, 0.8, 2.0)
#kernel
io.save_image_colored(kernel, "output/NIRCam_F070W_PSF.png", "inferno", True, 0, 0.2)
#corrupted image
io.save_image_colored(noisy_blurry_photo, "output/NIRCam_F070W_conv.png", "inferno",  True, 0, 1)
io.save_image_colored(noisy_blurry_photo, "output/NIRCam_F070W_conv_neg.png", "gray_r",  True, 0, 1)
#deconvolved image
io.save_image_colored(deconv_image, "output/hubble_F210M_rl_decon.png", "inferno", False, 0, 0.8)
io.save_image_colored(deconv_image, "output/hubble_F210M_rl_decon_neg.png", "gray_r", False, 0, 0.8)


fig, axes = plt.subplots(2, 3, figsize = (15, 10))
    
axes[0, 0].imshow(original_image, cmap="magma")
axes[0, 0].set_title("Original")

axes[0, 1].imshow(kernel, cmap="magma")
axes[0, 1].set_title("JWST NIRCam PSF")

axes[0, 2].imshow(noisy_blurry_photo, cmap="inferno")
axes[0, 2].set_title("Simulated JSWT Observation")

axes[1, 0].imshow(deconv_image, cmap="inferno")


plt.tight_layout()
plt.show()    