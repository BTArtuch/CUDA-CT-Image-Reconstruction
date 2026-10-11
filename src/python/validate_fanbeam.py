import os
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray):
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    mae = np.mean(np.abs(original - reconstructed))
    return rmse, mae

def validate_fanbeam(size: int):
    phantom_path = f"data/phantom_{size}.bin"
    cuda_path = f"results/recon_cuda_fanbeam_{size}.bin"
    
    if not (os.path.exists(phantom_path) and os.path.exists(cuda_path)):
        print(f"Missing data for {size}x{size}.")
        return

    phantom = np.fromfile(phantom_path, dtype=np.float32).reshape((size, size))
    recon_cuda = np.fromfile(cuda_path, dtype=np.float32).reshape((size, size))

    # Computation is natively scaled now; no manual D/2.0 correction needed
    rmse, mae = compute_metrics(phantom, recon_cuda)

    print(f"--- {size}x{size} Fan-Beam Validation ---")
    print(f"CUDA Fan-Beam RMSE: {rmse:.5f} | MAE: {mae:.5f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Original Phantom ({size}x{size})")
    axes[0].axis('off')
    
    axes[1].imshow(recon_cuda, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"CUDA Fan-Beam Recon\nRMSE: {rmse:.5f}")
    axes[1].axis('off')
    
    out_path = f"results/validation_fanbeam_{size}x{size}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"Validation plot saved to: {out_path}")

if __name__ == "__main__":
    for res in [128, 256, 512]:
        validate_fanbeam(res)