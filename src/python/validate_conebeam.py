import os
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray):
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    mae = np.mean(np.abs(original - reconstructed))
    return rmse, mae

def validate_conebeam(size: int):
    f_p_center = f"data/phantom_cone_center_{size}.bin"
    f_r_center = f"results/recon_cone_center_{size}.bin"
    f_p_edge = f"data/phantom_cone_edge_{size}.bin"
    f_r_edge = f"results/recon_cone_edge_{size}.bin"
    
    if not all(os.path.exists(f) for f in [f_p_center, f_r_center, f_p_edge, f_r_edge]):
        print(f"Missing binaries for {size}^3. Run fbp_conebeam_cuda first.")
        return

    # Load Slices
    p_center = np.fromfile(f_p_center, dtype=np.float32).reshape((size, size))
    r_center = np.fromfile(f_r_center, dtype=np.float32).reshape((size, size))
    p_edge = np.fromfile(f_p_edge, dtype=np.float32).reshape((size, size))
    r_edge = np.fromfile(f_r_edge, dtype=np.float32).reshape((size, size))

    rmse_c, mae_c = compute_metrics(p_center, r_center)
    rmse_e, mae_e = compute_metrics(p_edge, r_edge)

    print(f"--- {size}^3 Cone-Beam Validation ---")
    print(f"Center Slice RMSE: {rmse_c:.5f} | MAE: {mae_c:.5f} (Optimal Data)")
    print(f"Edge Z-Slice RMSE: {rmse_e:.5f} | MAE: {mae_e:.5f} (Tuy's Condition Violated)\n")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"3D Cone-Beam Artifact Validation ({size}^3)", fontsize=16)

    axes[0,0].imshow(p_center, cmap='gray', vmin=0, vmax=1)
    axes[0,0].set_title("Phantom: Center Slice")
    axes[0,1].imshow(r_center, cmap='gray', vmin=0, vmax=1)
    axes[0,1].set_title(f"Recon: Center Slice (RMSE: {rmse_c:.4f})")

    axes[1,0].imshow(p_edge, cmap='gray', vmin=0, vmax=1)
    axes[1,0].set_title("Phantom: Edge Slice (85% Z)")
    axes[1,1].imshow(r_edge, cmap='gray', vmin=0, vmax=1)
    axes[1,1].set_title(f"Recon: Edge Slice (RMSE: {rmse_e:.4f})\nNotice Vertical Blurring")

    for ax in axes.flat: ax.axis('off')
    
    out_path = f"results/validation_conebeam_{size}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    for res in [128, 256, 512]:
        validate_conebeam(res)