import numpy as np
import matplotlib.pyplot as plt
import deconv_utils as dcu
import image_io as io 


image_size = (256, 256)

original_image = dcu.create_image(shape="circle", size=image_size, shape_size=50)

kernel = dcu.create_kernel(ktype = "gaussian", ksize = 15, sigma = 5)

H = dcu.get_otf(kernel, target_size=image_size)

F_image = np.fft.fft2(original_image)

F_blur = F_image * H

blurry_image = np.real(np.fft.ifft2(F_blur))

blur_noise_image = dcu.noise_poission_gaussian(blurry_image, 2000, 0.005)
F_bn = np.fft.fft2(blur_noise_image)


plt.figure()
plt.imshow(blurry_image, cmap = "gray")
plt.title("blurry image")
plt.show()

deconv_brute = dcu.deconv_brute(F_bn, H, epsilon=1e-9)

plt.figure()
plt.imshow(deconv_brute, cmap = "gray")
plt.title("deconv brute")
plt.show()

deconv_wiener = dcu.deconv_wiener(F_bn, H, K = 10)

plt.figure()
plt.imshow(deconv_wiener, cmap = "gray")
plt.title("deconv wiener")
plt.show()

deconv_rl = dcu.deconv_rl(blur_noise_image, H ,iteration=30)

plt.figure()
plt.imshow(deconv_rl, cmap = "gray", vmin = 0, vmax = 1)
plt.title("deconv rl")
plt.show()

deconv_rl_fix = dcu.deconv_rl_damped(blur_noise_image, H, 30, 0.001)

plt.figure()
plt.imshow(deconv_rl_fix, cmap="gray", vmin = 0, vmax = 1)
plt.title("deconv rl fix")
plt.show()