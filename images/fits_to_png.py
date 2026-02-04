import numpy as np
import sys
import os
from astropy.io import fits # Requires: pip install astropy
from PIL import Image

def convert_fits_to_png(fits_path):
    """
    Reads a FITS file and saves:
    1. A .NPY file (Raw Data) -> Use this in your python code!
    2. A .PNG file (Log View) -> Use this to look at the snowflake.
    """
    filename = os.path.splitext(os.path.basename(fits_path))[0]
    
    try:
        # 1. Load FITS Data
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            if data is None:
                data = hdul[1].data
        
        # Handle 3D stacks
        while data.ndim > 2:
            data = data[0]
            
        # Cleanup
        data = np.nan_to_num(data)
        data = np.clip(data, 0, None)
        
        # --- SAVE 1: THE RAW MATH DATA (.NPY) ---
        # This keeps the original massive dynamic range for your math.
        npy_name = f"{filename}.npy"
        np.save(npy_name, data)
        print(f"  -> Saved RAW MATH data: {npy_name} (USE THIS IN CODE)")

        # --- SAVE 2: THE VISUAL PREVIEW (.PNG) ---
        # Normalize to max for visualization
        d_max = np.max(data)
        if d_max > 0:
            data_norm = data / d_max
        else:
            data_norm = data
            
        # SAFETY: Hard Noise Gate
        # Force the deep background to 0.0 to prevent "explosions" in convolution
        # caused by faint gray fog.
        data_norm[data_norm < 1e-5] = 0.0
            
        # Log Stretch (Makes wings visible)
        boost = 100000
        data_log = np.log10(1 + boost * data_norm) / np.log10(1 + boost)
        
        img_uint8 = (data_log * 255).astype(np.uint8)
        png_name = f"{filename}.png"
        Image.fromarray(img_uint8).save(png_name)
        print(f"  -> Saved VISUAL preview: {png_name} (Cleaned background)")

    except Exception as e:
        print(f"Error converting {fits_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for file_path in sys.argv[1:]:
            convert_fits_to_png(file_path)
    else:
        print("Usage: python fits_to_png.py <file.fits>")