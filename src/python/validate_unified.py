import os
import numpy as np

def compute_rmse(original: np.ndarray, reconstructed: np.ndarray):
    return np.sqrt(np.mean((original - reconstructed) ** 2))

def print_markdown_table():
    print("| Resolution | Parallel RMSE | Fan-Beam RMSE | Cone-Beam RMSE |")
    print("| :--- | :--- | :--- | :--- |")
    
    for size in [128, 256, 512]:
        phantom_path = f"data/phantom_cone_center_{size}.bin"
        par_path = f"results/recon_cuda_{size}.bin"
        fan_path = f"results/recon_cuda_fanbeam_{size}.bin"
        cone_path = f"results/recon_cone_center_{size}.bin"
        
        if not all(os.path.exists(p) for p in [phantom_path, par_path, fan_path, cone_path]):
            print(f"| **{size}x{size}** | Missing Data | Missing Data | Missing Data |")
            continue
            
        phantom = np.fromfile(phantom_path, dtype=np.float32).reshape((size, size))
        par_recon = np.fromfile(par_path, dtype=np.float32).reshape((size, size))
        fan_recon = np.fromfile(fan_path, dtype=np.float32).reshape((size, size))
        cone_recon = np.fromfile(cone_path, dtype=np.float32).reshape((size, size))
        
        par_rmse = compute_rmse(phantom, par_recon)
        fan_rmse = compute_rmse(phantom, fan_recon)
        cone_rmse = compute_rmse(phantom, cone_recon)
        
        print(f"| **{size}^3** | {par_rmse:.5f} | {fan_rmse:.5f} | {cone_rmse:.5f} |")

if __name__ == "__main__":
    print("\n--- Unified CUDA RMSE Benchmarks ---")
    print_markdown_table()
    print("\n")