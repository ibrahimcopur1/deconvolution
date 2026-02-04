import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(filepath):
    try:
        #.npy file support
        if filepath.endswith('.npy'):
            img_array = np.load(filepath)
            img_array = np.nan_to_num(img_array)
            print(f"Loaded RAW NPY '{filepath}': Shape {img_array.shape}, Max {np.max(img_array):.2e}")
            return img_array.astype(np.float64)

        #JPG/PNG support
        img = Image.open(filepath)
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        img_float = img_array.astype(np.float64) / 255.0
        
        print(f"Loaded '{filepath}': Shape {img_float.shape}")
        return img_float
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_kernel(filepath):

    kernel = load_image(filepath)
    if kernel is None:
        return None

    #normalization
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel = kernel / kernel_sum
    else:
        print("  ERROR: Kernel is empty (sum=0).")
    
    print(f"Loaded Kernel: Sum={np.sum(kernel):.2f}, Peak={np.max(kernel):.2e}")
    return kernel

def save_image(img_array, filepath):
    try:
        img_array = np.nan_to_num(img_array)
        img_clipped = np.clip(img_array, 0.0, 1.0)
        img_uint8 = (img_clipped * 255.0).astype(np.uint8)
        result_img = Image.fromarray(img_uint8)
        result_img.save(filepath)
        print(f"Saved result to '{filepath}'")
    except Exception as e:
        print(f"Error saving image: {e}")

def save_image_colored(img_array, filepath, cmap='magma', log=False, threshold=0.0, power=1.0):
    
    try:
        #cleaning up
        img_clean = np.nan_to_num(img_array)
        
        #we first log-compress THEN clip the values at 1.0, otherwise we would lose the huge dynamic
        #range between the stars.

        if log:
            if np.min(img_clean) < 0:
                img_clean = img_clean - np.min(img_clean)
                
            img_clean = np.log1p(img_clean)
            
            #normalize on the logged image
            m = np.max(img_clean)
            if m > 0:
                img_clean = img_clean / m
        
        #clipping
        img_clipped = np.clip(img_clean, 0.0, 1.0)
        
        #applying a threshold to zero out any smears under the threshold value
        if threshold > 0:
            img_clipped[img_clipped < threshold] = 0.0
            
        #dimming the halos
        if power != 1.0:
            img_clipped = img_clipped ** power

        #colormap
        colormap_func = plt.get_cmap(cmap)
        img_rgba = colormap_func(img_clipped)
        
       
        img_rgb = (img_rgba[:, :, :3] * 255).astype(np.uint8)
        
      
        Image.fromarray(img_rgb).save(filepath)
        
        print(f"Saved COLORED result to '{filepath}' (Shape: {img_rgb.shape})")
        
    except Exception as e:
        print(f"Error saving colored image: {e}")     