import os
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray):
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    mae = np.mean(np.abs(original - reconstructed))
    return rmse, mae

def validate_and_plot(size: int):
    phantom_path = f"data/phantom_{size}.bin"
    cpp_path = f"results/recon_cpp_{size}.bin"
    cuda_path = f"results/recon_cuda_{size}.bin"
    
    if not (os.path.exists(phantom_path) and os.path.exists(cpp_path) and os.path.exists(cuda_path)):
        print(f"Missing data for {size}x{size}. Did you run all benchmarks?")
        return

    # Load raw binaries into 2D arrays
    phantom = np.fromfile(phantom_path, dtype=np.float32).reshape((size, size))
    recon_cpp = np.fromfile(cpp_path, dtype=np.float32).reshape((size, size))
    recon_cuda = np.fromfile(cuda_path, dtype=np.float32).reshape((size, size))

    # Compute metrics against the original phantom
    cpp_rmse, cpp_mae = compute_metrics(phantom, recon_cpp)
    cuda_rmse, cuda_mae = compute_metrics(phantom, recon_cuda)

    # Output to console
    print(f"--- {size}x{size} Validation ---")
    print(f"C++  RMSE: {cpp_rmse:.5f} | MAE: {cpp_mae:.5f}")
    print(f"CUDA RMSE: {cuda_rmse:.5f} | MAE: {cuda_mae:.5f}\n")

    # Generate comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Original Phantom ({size}x{size})")
    axes[0].axis('off')
    
    axes[1].imshow(recon_cpp, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"C++ Reconstruction\nRMSE: {cpp_rmse:.5f}")
    axes[1].axis('off')
    
    axes[2].imshow(recon_cuda, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"CUDA Reconstruction\nRMSE: {cuda_rmse:.5f}")
    axes[2].axis('off')
    
    out_path = f"results/validation_compare_{size}x{size}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    for res in [128, 256, 512]:
        validate_and_plot(res)