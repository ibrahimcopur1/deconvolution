import numpy as np 


#image and kernel creation

def create_image(shape='circle', size=(256, 256), shape_size=50):
    original_image = np.zeros(size)

    if shape == "square":
        start = (size[0] - shape_size) // 2
        end = start + shape_size
        original_image[start:end, start:end] = 1.0

    elif shape == "circle":
        cx, cy = size[0] // 2, size[1] // 2
        x = np.arange(0, size[1])
        y = np.arange(0, size[0])
        xx, yy = np.meshgrid(x, y)
        distance = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        original_image[distance <= shape_size] = 1.0

    elif shape == "grad":
        linspace = np.linspace(0.0, 1.0, size[0])

        xx, yy = np.meshgrid(linspace, linspace)

        original_image = yy    

    return original_image        

def create_kernel(ktype='gaussian' , ksize=15, sigma=4):
    kernel_center = ksize // 2

    if ktype == 'gaussian':
        x = np.linspace(-kernel_center, kernel_center, ksize)
        xx, yy = np.meshgrid(x ,x)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    elif ktype == 'motion':
        kernel = np.zeros((ksize, ksize))
        kernel[kernel_center, : ] = 1.0

    elif ktype == 'box':
        kernel = np.ones((ksize, ksize))

    #normalize the kernels!
    return kernel / np.sum(kernel)

#noise generation
def add_noise(image, strength = 0.01):
    noise = np.random.normal(0, strength, size = image.shape)
    return image + noise

def noise_poission_gaussian(image, iso = 1000, read_noise = 0.005):

    image = np.real(image)
    image = np.nan_to_num(image)
    image = np.clip(image, 0, None)

    image_photons = image*iso
    noisy_photons = np.random.poisson(image_photons).astype(float)
    image_shot_noise = noisy_photons / iso
    electronic_noise = np.random.normal(0, read_noise, size=image.shape)

    final_image = image_shot_noise + electronic_noise

    return np.clip(final_image, 0.0, 1.0)




#fft and padding 
def get_otf(kernel, target_size):
    k_size = kernel.shape[0]

    padded_kernel = np.zeros(target_size)

    start_pad = (target_size[0] - k_size) // 2
    end_pad = start_pad + k_size
    padded_kernel[start_pad:end_pad, start_pad:end_pad] = kernel

    rolled_padded_kernel = np.fft.ifftshift(padded_kernel)

    H = np.fft.fft2(rolled_padded_kernel)
    return H

def get_laplacian_fft(target_size):
    p = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ])

    padded_laplacian = np.zeros(target_size)
    start_lap = (target_size[0] - 3) // 2
    end_lap = start_lap + 3
    padded_laplacian[start_lap:end_lap, start_lap:end_lap] = p

    rolled_padded_laplacian = np.fft.ifftshift(padded_laplacian)

    P = np.fft.fft2(rolled_padded_laplacian)
    return P


def deconv_brute(F_blur_noisy, H, epsilon = 1e-9):
    F_deconv = F_blur_noisy / H
    return np.real(np.fft.ifft2(F_deconv))

def deconv_wiener(F, H, K = 0.1):
    H_conj = np.conjugate(H) 
    H_mag_sq = np.abs(H)**2

    W = H_conj / (H_mag_sq + K)

    F_deconv = F * W
    return np.real(np.fft.ifft2(F_deconv))

def deconv_rl(image, H, iteration = 30):
    #takes the original image not the freq. domain F 
    H_conj = np.conjugate(H)
    f = image.copy()
    epsilon = 1e-12

    for i in range(iteration):
        F = np.fft.fft2(f)
        F_reblur = F * H
        g = np.real(np.fft.ifft2(F_reblur))

        g[g <= epsilon] = epsilon
        ratio = image / g

        F_ratio = np.fft.fft2(ratio)
        F_corr = F_ratio * H_conj
        corr = np.real(np.fft.ifft2(F_corr))

        f = f * corr

        f[f < 0] = 0

    return f    

#fixing rl artifacts
def deconv_rl_damped(noisy_blurry_image, H, iteration = 30, threshold = 0.02):
    #stops updating pixels when difference is below threshold to prevent noise amplifications
    
    H_conj = np.conjugate(H)
    f_k = noisy_blurry_image.copy()
    epsilon = 1e-10
   
    for i in range(iteration):
        F_k = np.fft.fft2(f_k)
        g_k = np.real(np.fft.ifft2(F_k * H))

        #damping
        diff = noisy_blurry_image - g_k
        skip = np.abs(diff) < threshold

        g_k[g_k <= epsilon] = epsilon
        ratio = noisy_blurry_image / g_k

        #force ratio for the skipped pixels
        ratio[skip] = 1

        correction = np.real(np.fft.ifft2(np.fft.fft2(ratio) * H_conj))

        f_k = f_k * correction
        f_k[f_k < 0] = 0

        return f_k

def create_jwst_psf(size=256, strut_thickness=0.2): 
    aperture_size = size * 2
    cy, cx = aperture_size // 2, aperture_size // 2
    y, x = np.mgrid[-cy:aperture_size-cy, -cx:aperture_size-cx]

    y = y / cy
    x = x / cx

    radius = 0.12

    sin60 = np.sin(np.pi/3)
    cos60 = np.cos(np.pi/3)


    hex_mask = (np.abs(y) < radius) & \
               (np.abs(x * sin60 + y * cos60) < radius) & \
               (np.abs(x * -sin60 + y * cos60) < radius)

    aperture = hex_mask.astype(float)

    strut_mask = (np.abs(x) < strut_thickness) & (y < 0) & (y > -radius)
    aperture[strut_mask] = 0 #struts block light

    strut_left = (np.abs(x * -sin60 + y * cos60) < strut_thickness) & (y > 0)
    aperture[strut_left & hex_mask] = 0

    strut_right = (np.abs(x * -sin60 + y * cos60) < strut_thickness) & (y > 0)
    aperture[strut_right & hex_mask] = 0.0


    F_aperture = np.fft.fft2(np.fft.ifftshift(aperture))
    F_shifted = np.fft.fftshift(F_aperture)
    psf = np.abs(F_shifted) ** 2

    #cropping
    start = (aperture_size - size) // 2
    psf_cropped = psf[start:start+size, start:start+size]

    return psf_cropped / np.sum(psf_cropped)








