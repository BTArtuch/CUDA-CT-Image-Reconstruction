import os
import numpy as np

def generate_unified_datasets(size: int):
    phantom_path = f"data/phantom_cone_center_{size}.bin"
    if not os.path.exists(phantom_path):
        print(f"Missing {phantom_path}. Run the Cone-Beam CUDA pipeline first.")
        return
        
    print(f"--- Generating Unified {size}x{size} Datasets ---")
    phantom = np.fromfile(phantom_path, dtype=np.float32).reshape((size, size))
    
    # Grid Setup
    center = (size - 1) / 2.0
    x = np.arange(size) - center
    y = np.arange(size) - center
    X, Y = np.meshgrid(x, -y)
    
    # Filter Setup
    pad_len = int(2 ** np.ceil(np.log2(2 * size)))
    freqs = np.fft.fftfreq(pad_len)
    ramp = 2.0 * np.abs(freqs).astype(np.float32)

    # ==========================================
    # 1. Parallel-Beam (180 Angles)
    # ==========================================
    num_angles_par = 180
    angles_par = np.linspace(0, np.pi, num_angles_par, endpoint=False, dtype=np.float32)
    sino_par = np.zeros((num_angles_par, size), dtype=np.float32)
    
    for i, theta in enumerate(angles_par):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        T = X * cos_t + Y * sin_t + center
        
        t_low = np.floor(T).astype(int)
        t_high = t_low + 1
        d_high = T - t_low
        d_low = 1.0 - d_high
        
        v_low = (t_low >= 0) & (t_low < size)
        v_high = (t_high >= 0) & (t_high < size)
        
        np.add.at(sino_par[i], t_low[v_low], (phantom * d_low)[v_low])
        np.add.at(sino_par[i], t_high[v_high], (phantom * d_high)[v_high])
        
    sino_fft_par = np.fft.fft(sino_par, n=pad_len, axis=1)
    filt_par = np.fft.ifft(sino_fft_par * ramp, axis=1).real[:, :size].astype(np.float32)
    
    # FIX: Removed the incorrect d_theta multiplier here. The CUDA kernel already applies it.
    
    filt_par.tofile(f"data/filtered_sino_{size}_{num_angles_par}.bin")
    angles_par.tofile(f"data/angles_{num_angles_par}.bin")

    # ==========================================
    # 2. Fan-Beam (360 Angles)
    # ==========================================
    num_angles_fan = 360
    betas = np.linspace(0, 2 * np.pi, num_angles_fan, endpoint=False, dtype=np.float32)
    D = size * 1.5
    gamma_max = np.arcsin((size * 0.5 * np.sqrt(2)) / D)
    gammas = np.linspace(-gamma_max, gamma_max, size, dtype=np.float32)
    d_gamma = gammas[1] - gammas[0]
    gamma_min = gammas[0]
    
    sino_fan = np.zeros((num_angles_fan, size), dtype=np.float32)
    
    for i, beta in enumerate(betas):
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        s = X * cos_b + Y * sin_b
        t = -X * sin_b + Y * cos_b
        
        gamma_pixels = np.arctan2(s, D + t)
        
        gamma_bin = (gamma_pixels - gamma_min) / d_gamma
        g_low = np.floor(gamma_bin).astype(int)
        g_high = g_low + 1
        w_high = gamma_bin - g_low
        w_low = 1.0 - w_high
        
        v_low = (g_low >= 0) & (g_low < size)
        v_high = (g_high >= 0) & (g_high < size)
        
        # FIX: Removed the broken div_weight splatting calculation.
        np.add.at(sino_fan[i], g_low[v_low], (phantom * w_low)[v_low])
        np.add.at(sino_fan[i], g_high[v_high], (phantom * w_high)[v_high])
        
    pre_weights = D * np.cos(gammas)
    weighted_sino = sino_fan * pre_weights
    
    sino_fft_fan = np.fft.fft(weighted_sino, n=pad_len, axis=1)
    filt_fan = np.fft.ifft(sino_fft_fan * ramp, axis=1).real[:, :size].astype(np.float32)
    
    # FIX: Correct amplitude scaling. 
    # Accounts for the CUDA 1/L^2 division and 2*pi integration range
    filt_fan *= (D * 0.5)
    
    filt_fan.tofile(f"data/fan_filtered_sino_{size}_{num_angles_fan}.bin")
    betas.tofile(f"data/fan_betas_{num_angles_fan}.bin")
    gammas.tofile(f"data/fan_gammas_{size}.bin")

if __name__ == "__main__":
    for res in [128, 256, 512]:
        generate_unified_datasets(res)