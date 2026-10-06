import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

def generate_phantom(size: int) -> np.ndarray:
    """Generate a 2D Shepp-Logan phantom scaled to size x size."""
    base_phantom = shepp_logan_phantom()
    scale_factor = size / base_phantom.shape[0]
    phantom = rescale(base_phantom, scale_factor, mode='reflect', anti_aliasing=True)
    return phantom.astype(np.float32)

def forward_project(image: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Compute parallel-beam forward projection (sinogram).
    image: (N, N)
    angles: (num_angles,) in radians
    Returns: sinogram (num_angles, num_detectors) where num_detectors = N
    """
    N = image.shape[0]
    num_angles = len(angles)
    sinogram = np.zeros((num_angles, N), dtype=np.float32)
    
    # Coordinate system centered at image center
    center = (N - 1) / 2.0
    x = np.arange(N) - center
    y = np.arange(N) - center
    X, Y = np.meshgrid(x, -y)  # Standard Cartesian orientation
    
    for i, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Detector coordinate for every pixel
        T = X * cos_t + Y * sin_t + center
        
        # Accumulate pixel values onto discrete bins
        t_low = np.floor(T).astype(int)
        t_high = t_low + 1
        d_high = T - t_low
        d_low = 1.0 - d_high
        
        valid_low = (t_low >= 0) & (t_low < N)
        valid_high = (t_high >= 0) & (t_high < N)
        
        np.add.at(sinogram[i], t_low[valid_low], (image * d_low)[valid_low])
        np.add.at(sinogram[i], t_high[valid_high], (image * d_high)[valid_high])
        
    return sinogram

def ram_lak_filter(sinogram: np.ndarray) -> np.ndarray:
    """
    Apply 1D Ram-Lak ramp filter in frequency domain along detector axis.
    sinogram: (num_angles, num_detectors)
    """
    num_angles, num_detectors = sinogram.shape
    padded_len = int(2 ** np.ceil(np.log2(2 * num_detectors)))
    
    # Frequency grid [-0.5, 0.5)
    freqs = np.fft.fftfreq(padded_len)
    ramp = 2.0 * np.abs(freqs).astype(np.float32)
    
    # 1D FFT along detector axis
    sino_fft = np.fft.fft(sinogram, n=padded_len, axis=1)
    filtered_fft = sino_fft * ramp[np.newaxis, :]
    filtered_sino = np.fft.ifft(filtered_fft, axis=1).real
    
    # Crop back to original detector length
    return filtered_sino[:, :num_detectors].astype(np.float32)

def backproject(filtered_sinogram: np.ndarray, angles: np.ndarray, image_size: int) -> np.ndarray:
    """
    Backproject filtered sinogram into image space.
    This structure directly reflects the parallel thread model for CUDA.
    """
    num_angles, num_detectors = filtered_sinogram.shape
    center = (image_size - 1) / 2.0
    det_center = (num_detectors - 1) / 2.0
    
    x = np.arange(image_size) - center
    y = np.arange(image_size) - center
    X, Y = np.meshgrid(x, -y)
    
    reconstruction = np.zeros((image_size, image_size), dtype=np.float32)
    d_theta = np.pi / num_angles
    
    for i, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Compute detector intersection t for each pixel (x, y)
        t = X * cos_t + Y * sin_t + det_center
        
        # Linear interpolation sampling along the detector
        t0 = np.floor(t).astype(int)
        t1 = t0 + 1
        wt1 = t - t0
        wt0 = 1.0 - wt1
        
        mask = (t0 >= 0) & (t1 < num_detectors)
        proj_slice = filtered_sinogram[i]
        
        reconstruction[mask] += (proj_slice[t0[mask]] * wt0[mask] +
                                 proj_slice[t1[mask]] * wt1[mask])
        
    return reconstruction * d_theta

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray):
    """Calculate Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    mae = np.mean(np.abs(original - reconstructed))
    return rmse, mae

def run_benchmark(size: int = 256, num_angles: int = 180):
    print(f"--- Running Python FBP Benchmark: {size}x{size}, {num_angles} angles ---")
    phantom = generate_phantom(size)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False, dtype=np.float32)
    
    # 1. Forward projection timing
    t0 = time.perf_counter()
    sinogram = forward_project(phantom, angles)
    t1 = time.perf_counter()
    t_fwd = (t1 - t0) * 1000.0
    
    # 2. Ramp filter timing
    t0 = time.perf_counter()
    filtered_sino = ram_lak_filter(sinogram)
    t1 = time.perf_counter()
    t_filt = (t1 - t0) * 1000.0
    
    # 3. Backprojection timing
    t0 = time.perf_counter()
    reconstruction = backproject(filtered_sino, angles, size)
    t1 = time.perf_counter()
    t_back = (t1 - t0) * 1000.0
    
    total_time = t_fwd + t_filt + t_back
    rmse, mae = compute_metrics(phantom, reconstruction)
    
    print(f"Forward Projection: {t_fwd:8.2f} ms")
    print(f"Ramp Filtering:     {t_filt:8.2f} ms")
    print(f"Backprojection:     {t_back:8.2f} ms")
    print(f"Total Pipeline:     {total_time:8.2f} ms")
    print(f"Reconstruction RMSE: {rmse:.5f} | MAE: {mae:.5f}")
    
    # Save visual confirmation
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(phantom, cmap='gray')
    axes[0].set_title(f"Original Phantom ({size}x{size})")
    axes[0].axis('off')
    
    axes[1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1].set_title(f"Sinogram ({num_angles} x {size})")
    axes[1].set_xlabel("Detector Bin")
    axes[1].set_ylabel("Angle Index")
    
    axes[2].imshow(reconstruction, cmap='gray')
    axes[2].set_title(f"FBP Reconstruction\nRMSE: {rmse:.4f}")
    axes[2].axis('off')
    
    out_path = f"results/python_baseline_{size}x{size}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Validation plot saved to: {out_path}\n")

    os.makedirs("data", exist_ok=True)
    filtered_sino.tofile(f"data/filtered_sino_{size}_{num_angles}.bin")
    angles.tofile(f"data/angles_{num_angles}.bin")
    phantom.tofile(f"data/phantom_{size}.bin")

if __name__ == "__main__":
    # Standard benchmark sweeps
    for resolution in [128, 256, 512]:
        run_benchmark(size=resolution, num_angles=180)