import os
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

def generate_phantom(size: int) -> np.ndarray:
    base_phantom = shepp_logan_phantom()
    scale_factor = size / base_phantom.shape[0]
    return rescale(base_phantom, scale_factor, mode='reflect', anti_aliasing=True).astype(np.float32)

def forward_project_fanbeam(image: np.ndarray, betas: np.ndarray, gammas: np.ndarray, D: float) -> np.ndarray:
    N = image.shape[0]
    num_angles = len(betas)
    num_detectors = len(gammas)
    sinogram = np.zeros((num_angles, num_detectors), dtype=np.float32)

    center = (N - 1) / 2.0
    x = np.arange(N) - center
    y = np.arange(N) - center
    X, Y = np.meshgrid(x, -y)

    d_gamma = gammas[1] - gammas[0]
    gamma_min = gammas[0]

    for i, beta in enumerate(betas):
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)

        s = X * cos_b + Y * sin_b
        t = -X * sin_b + Y * cos_b

        # Calculate distance (L) and divergent angle (gamma)
        L = np.sqrt(s**2 + (D + t)**2)
        gamma_pixels = np.arctan2(s, D + t)

        gamma_bin = (gamma_pixels - gamma_min) / d_gamma
        gamma_low = np.floor(gamma_bin).astype(int)
        gamma_high = gamma_low + 1
        w_high = gamma_bin - gamma_low
        w_low = 1.0 - w_high

        valid_low = (gamma_low >= 0) & (gamma_low < num_detectors)
        valid_high = (gamma_high >= 0) & (gamma_high < num_detectors)

        # DIVERGENCE WEIGHT: Compensates for the wider beam spreading at the back of the object
        divergence_weight = 1.0 / (L * d_gamma)

        np.add.at(sinogram[i], gamma_low[valid_low], (image * w_low * divergence_weight)[valid_low])
        np.add.at(sinogram[i], gamma_high[valid_high], (image * w_high * divergence_weight)[valid_high])

    return sinogram

def ram_lak_filter_fanbeam(sinogram: np.ndarray, gammas: np.ndarray, D: float) -> np.ndarray:
    pre_weights = D * np.cos(gammas)
    weighted_sino = sinogram * pre_weights[np.newaxis, :]

    num_angles, num_detectors = weighted_sino.shape
    padded_len = int(2 ** np.ceil(np.log2(2 * num_detectors)))

    freqs = np.fft.fftfreq(padded_len)
    ramp = 2.0 * np.abs(freqs).astype(np.float32)

    sino_fft = np.fft.fft(weighted_sino, n=padded_len, axis=1)
    filtered_fft = sino_fft * ramp[np.newaxis, :]
    filtered_sino = np.fft.ifft(filtered_fft, axis=1).real

    # Apply missing discrete sampling scale (1/d_gamma) and 360-degree double integration scale (0.5)
    d_gamma = gammas[1] - gammas[0]
    final_scale = 0.5 / d_gamma

    return (filtered_sino[:, :num_detectors] * final_scale).astype(np.float32)

def run_fanbeam_generation(size: int = 256, num_angles: int = 360):
    print(f"--- Generating Fan-Beam Data: {size}x{size} ---")
    phantom = generate_phantom(size)

    D = size * 1.5 
    betas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False, dtype=np.float32)

    gamma_max = np.arcsin((size * 0.5 * np.sqrt(2)) / D)
    gammas = np.linspace(-gamma_max, gamma_max, size, dtype=np.float32)

    sinogram = forward_project_fanbeam(phantom, betas, gammas, D)
    filtered_sino = ram_lak_filter_fanbeam(sinogram, gammas, D)

    os.makedirs("data", exist_ok=True)
    filtered_sino.tofile(f"data/fan_filtered_sino_{size}_{num_angles}.bin")
    betas.tofile(f"data/fan_betas_{num_angles}.bin")
    gammas.tofile(f"data/fan_gammas_{size}.bin")
    
    print(f"Exported {size}x{size} binaries to data/ \n")

if __name__ == "__main__":
    for res in [128, 256, 512]:
        run_fanbeam_generation(size=res, num_angles=360)