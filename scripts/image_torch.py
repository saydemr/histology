import numpy as np
import cv2
import pywt
from skimage import exposure, restoration, morphology, filters
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
import kornia as K
from kornia.enhance import equalize_clahe
import ptwt

"""
Virtual H&E Staining Pipeline (GPU Accelerated)
CS-leaning Title: Physics-Based Beer-Lambert Mapping for Fluorescence to Brightfield Image Translation

This script implements the Physics-Based Beer-Lambert Mapping.
It translates two-channel fluorescence images (TO-PRO-3 and Eusion) into a Virtual H&E brightfield image.
The implementation utilizes PyTorch for FFT, matrix operations, and color handling.
"""

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Defining the Color Vectors (Ruifrok and Johnston)
# Absorption values for [Red, Green, Blue]
V_H = torch.tensor([0.644, 0.717, 0.267], device=device, dtype=torch.float32)
V_E = torch.tensor([0.093, 0.954, 0.283], device=device, dtype=torch.float32)

# Default gain factors
k_H = 1.0
k_E = 1.0

def fft_damp(img_tensor, sigma=10):
    """
    Applies FFT damping using PyTorch on GPU.
    Input/Output are numpy arrays to interface with pywt/ptwt.
    """
    # Move to GPU
    rows, cols = img_tensor.shape
    
    # Transform to Fourier Domain
    f_transform = torch.fft.fftshift(torch.fft.fft2(img_tensor))
    
    # Create Gaussian Mask
    cy, cx = rows // 2, cols // 2
    
    # Create meshgrid on GPU
    y = torch.arange(rows, device=device)
    x = torch.arange(cols, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Gaussian damping mask centered at DC
    mask = 1 - torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    
    # Apply mask
    f_filtered = f_transform * mask
    
    # Inverse FFT
    res = torch.real(torch.fft.ifft2(torch.fft.ifftshift(f_filtered)))
    
    # Return to CPU/Numpy for pywt reconstruction
    return res


def rolling_ball_pytorch(img_tensor, radius=50):
    """
    Applies rolling ball background subtraction using Kornia.
    
    Args:
        img_tensor (torch.Tensor): Input image [B, C, H, W], normalized 0-1.
        radius (int): Radius of the rolling ball.
        
    Returns:
        torch.Tensor: The background-subtracted image (Top-Hat transform).
    """
    # 1. Create the non-flat "Ball" kernel (structuring element)
    # The kernel represents the height of the ball at each pixel.
    kernel_size = 2 * radius + 1
    X, Y = np.meshgrid(np.linspace(-radius, radius, kernel_size),
                       np.linspace(-radius, radius, kernel_size))
    
    # Calculate height of the ball: z = sqrt(r^2 - x^2 - y^2)
    # Pixels outside the ball radius should be -infinity (masked out)
    dist_squared = X**2 + Y**2
    valid_mask = dist_squared <= radius**2
    
    # Initialize kernel with -inf (for erosion) or similar logic. 
    # Kornia expects the structuring element relative values.
    # For a ball, the values are the height offsets.
    ball_height = np.sqrt(np.maximum(0, radius**2 - dist_squared))
    
    # Normalize height to match image scale if necessary, usually kept relative.
    # However, for standard "rolling ball", we often just need the shape.
    # If strictly mimicking 2D flat rolling ball (ImageJ default), use a flat disk:
    # kernel = torch.from_numpy(valid_mask.astype(np.float32)).to(img_tensor.device)
    
    # EXACT PARITY with skimage "ball" kernel (non-flat):
    kernel = torch.from_numpy(ball_height).float().to(img_tensor.device)
    
    # Mask out values outside the circle to avoid square artifacts
    kernel[~torch.from_numpy(valid_mask).to(img_tensor.device)] = -float('inf')

    # 2. Apply Top Hat Transform
    # TopHat(I) = Image - Opening(Image)
    # Opening(I) = Dilation(Erosion(I))
    
    # Kornia's top_hat performs this automatically
    # Note: 'engine="convolution"' is faster but 'unfold' is more stable for non-flat
    result = K.morphology.top_hat(img_tensor, kernel=kernel, engine='unfold')
    
    return result


def destripe_channel(image, wavelet='db10', level=None, sigma=10):
    """
    Removes directional stripe artifacts.
    """
    # if level is None:
    #     level = pywt.dwt_max_level(min(image.shape), pywt.Wavelet(wavelet))
        # level = min(level, 5)

    # 1. Decomposition (CPU - pywt)
    # coeffs = pywt.wavedec2(image, wavelet, level=level)
    # print("Starting destriping using hybrid CPU-GPU approach...")
    coeffs_list = ptwt.wavedec2(image, wavelet, mode="symmetric", level=level)
    # coeffs_list = list(coeffs)
    
    new_list = [coeffs_list[0]]
    # 2. Filtering Detail Coefficients (GPU - fft_damp)
    for i in range(1, len(coeffs_list)):
        (cH, cV, cD) = coeffs_list[i]
        
        # We filter cV (Vertical Detail) and cH (Horizontal Detail)
        cV_filtered = fft_damp(cV, sigma)
        cH_filtered = fft_damp(cH, sigma)
        
        print(f"Level {i}: cH and cV filtered.")
        print(coeffs_list[i])
        new_list.append(ptwt.WaveletDetailTuple2d(cH_filtered, cV_filtered, cD))
        print(f"Level {i} coefficients updated.")

    # 3. Reconstruction (CPU - pywt)
    destriped_image = ptwt.waverec2(new_list, wavelet)
    
    # Clip to original range
    destriped_image = torch.clip(destriped_image, 0, torch.max(image))
    return destriped_image

def preprocess_channel(image, is_nuclear=True):
    """
    Destriping (GPU), CLAHE (CPU), Normalization (GPU)
    """
    # 1. Destriping (Hybrid)
    if min(image.shape) > 64:
        img = destripe_channel(image)
    
    # 2. Background Subtraction (CPU - skimage/scipy)
    # Keeping this on CPU as standard PyTorch lacks 'rolling_ball' equivalent
    background = restoration.rolling_ball(img, radius=50)
    img = img - background
    img = torch.maximum(img, 0)
    print("Background subtraction completed.")


    # 3. CLAHE (CPU - skimage)
    # Keeping this on CPU as standard PyTorch lacks adaptive histogram equalization
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=0.01, kernel_size=(64, 64))

    # 3. CLAHE (GPU - kornia)
    # kornia_clahe = equalize_clahe(img.unsqueeze(0).unsqueeze(0), clip_limit=0.01, tile_grid_size=(8,8))
    # img_clahe_kornia = kornia_clahe.squeeze().cpu().numpy()
    
    t_img = torch.from_numpy(img_clahe).to(device, dtype=torch.float32)

    # 4. Handling "Eusion" Noise Floor (GPU)
    if not is_nuclear:
        # Calculate 10th percentile, flatten for quantile
        flat = t_img.view(-1)
        k = int(0.1 * flat.numel())
        val, _ = torch.kthvalue(flat, k)
        noise_floor = val
        t_img = torch.maximum(t_img - noise_floor, torch.tensor(0.0, device=device))

    # 5. Percentile Scaling / Normalization
    flat = t_img.view(-1)
    k_999 = int(0.999 * flat.numel())
    k_999 = min(k_999, flat.numel() - 1)
    
    # Torch quantile/kthvalue for sorting
    # sort is safer for high percentiles if exact quantile not available in older torch
    sorted_vals, _ = torch.sort(flat)
    p99_9 = sorted_vals[k_999]

    if p99_9 > 0:
        img_final = t_img / p99_9
    else:
        img_final = t_img
        
    return torch.clamp(img_final, 0, 1.0)

def beer_lambert_mapping(nuc_img, cyto_img):
    """
    Color Mapping Algorithm via Beer-Lambert Law (Fully GPU)
    Inputs are PyTorch tensors.
    """
    H, W = nuc_img.shape
    
    # 1. Calculate Optical Density (OD)
    # Stack along last dim to mimic (H, W, 3)
    # OD_c = (N * k_H * V_H_c) + (C * k_E * V_E_c)
    
    # Expand dims for broadcasting: (H, W, 1) * (3,)
    nuc_expanded = nuc_img.unsqueeze(-1)
    cyto_expanded = cyto_img.unsqueeze(-1)
    
    OD = (nuc_expanded * k_H * V_H) + (cyto_expanded * k_E * V_E)
        
    # 2. Calculate Transmission
    # I = I0 * exp(-OD)
    transmission = 255.0 * torch.exp(-OD)
    
    # Clip to valid range
    transmission = torch.clamp(transmission, 0, 255).to(torch.uint8)
    
    return transmission

def post_process(image_tensor):
    """
    Color Balancing & Sharpening (Fully GPU)
    Input: (H, W, 3) Uint8 Tensor
    """
    img_float = image_tensor.to(torch.float32)
    
    # 1. Warmth Adjustment
    # R channel is index 0, B is index 2
    img_float[:, :, 0] *= 1.05 
    img_float[:, :, 2] *= 0.95 
    img_float = torch.clamp(img_float, 0, 255)
    
    # 2. Sharpening (Unsharp Mask)
    # Need (B, C, H, W) for grid-based ops or convolution
    # Permute to (C, H, W) and add batch dim
    img_permuted = img_float.permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)
    
    # Normalize to 0-1 for processing
    img_norm = img_permuted / 255.0
    
    # Gaussian Blur (radius 1.0 approx sigma 1.0)
    blurred = TF.gaussian_blur(img_norm, kernel_size=[5, 5], sigma=[1.0, 1.0])
    
    # Unsharp Mask: Original + (Original - Blurred) * Amount
    amount = 0.5
    sharpened = img_norm + (img_norm - blurred) * amount
    
    # Convert back
    sharpened = torch.clamp(sharpened * 255.0, 0, 255)
    
    # Remove batch dim and permute back to (H, W, 3)
    final_img = sharpened.squeeze(0).permute(1, 2, 0).to(torch.uint8)
    
    return final_img

def process_image_pair(to_pro_3_path: str, eusion_path: str, output_path=None):
    print(f"Loading images...\nNuclear: {to_pro_3_path}\nCytoplasmic: {eusion_path}")
    
    nuc_raw = torch.from_numpy(cv2.imread(to_pro_3_path, cv2.IMREAD_GRAYSCALE)).double().cuda()
    cyto_raw = torch.from_numpy(cv2.imread(eusion_path, cv2.IMREAD_GRAYSCALE)).double().cuda()

    print("Preprocessing Nuclear Channel (TO-PRO-3)...")
    # Returns GPU Tensor
    nuc_processed = preprocess_channel(nuc_raw, is_nuclear=True)
    
    print("Preprocessing Cytoplasmic Channel (Eusion)...")
    # Returns GPU Tensor
    cyto_processed = preprocess_channel(cyto_raw, is_nuclear=False)
    
    print("Applying Beer-Lambert Color Mapping...")
    # Returns GPU Tensor
    virtual_he = beer_lambert_mapping(nuc_processed, cyto_processed)

    print("Post-processing...")
    # Returns GPU Tensor
    final_he = post_process(virtual_he)

    # Move to CPU for saving
    final_he_cpu = final_he.cpu().numpy()

    # Convert RGB to BGR for OpenCV saving
    final_bgr = cv2.cvtColor(final_he_cpu, cv2.COLOR_RGB2BGR)
    
    if output_path:
        cv2.imwrite(output_path, final_bgr)
        print(f"Saved Virtual H&E to {output_path}")
        
    return final_he_cpu


nuc_image_dir = Path("./data/C01/")
cyto_image_dir = Path("./data/C02/")
output_dir = Path("./data/gpu_accelerated/")

output_dir.mkdir(parents=True, exist_ok=True)


try:
    for nuc_path in nuc_image_dir.glob("*.tif"):
        cyto_path = cyto_image_dir / nuc_path.name
        output_path = output_dir / nuc_path.name
        
        if not cyto_path.exists():
            print(f"Skipping {nuc_path.name}: Corresponding cytoplasmic image not found.")
            continue
        
        result = process_image_pair(str(nuc_path), str(cyto_path), str(output_path))
    
except Exception as e:
    print(f"Error: {e}")