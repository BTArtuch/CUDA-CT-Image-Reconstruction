# CT FBP Image Reconstruction Performance Study

## Setup
* OS: Ubuntu 26.04
* GPU: Nvidia GTX 750 Ti
* CUDA Toolkit: 12.4
* Target Architecture: sm_50
* Build Tools: CMake

### Relevant Commands

```bash
nvidia-smi
nvcc --version
nvcc --list-gpu-arch | grep 50
nvcc --list-gpu-code | grep 50
```

## Python Baseline

The Python baseline establishes the mathematical ground truth for the CUDA-CT-Reconstruction file using NumPy's vectorized operations. Crucially, it also acts as the data generator for the subsequent C++ and CUDA pipelines by exporting the intermediate mathematical steps as flat binary files, allowing us to isolate and profile strictly the backprojection algorithm later.

### Environment Setup

Ensure your virtual environment is active and install the baseline dependencies:

```bash
source venv/bin/activate
pip install numpy scipy matplotlib scikit-image
```

### Execution

Run the baseline script from the root of the repository:

```bash
python src/python/fbp_baseline.py
```

### Pipeline Steps & Outputs

When executed, the script automatically sweeps through three benchmark resolutions (128x128, 256x256, 512x512) and performs the following operations:

*   **Generation:** Creates a standard synthetic Shepp-Logan phantom.
*   **Forward Projection:** Simulates parallel X-ray beams across 180 projection angles to create a sinogram.
*   **Filtering:** Applies a 1D Ram-Lak ramp filter to the sinogram in the frequency domain.
*   **Backprojection & Validation:** Reconstructs the image and computes Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) against the original phantom.
*   **Visualization Export:** Saves diagnostic visual confirmation plots (e.g., `python_baseline_512x512.png`) to the `results/` directory.
*   **Binary Data Export:** Saves the raw phantom, the projection angles, and the filtered sinogram as `.bin` files into the `data/` directory.

### Reference Benchmarks

Recorded execution times for the pure Python CPU benchmark:

| Resolution | Forward Projection | Ramp Filtering | Backprojection | Total Pipeline | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **128x128** | 247.02 ms | 0.43 ms | 28.39 ms | 275.84 ms | 0.20102 |
| **256x256** | 992.84 ms | 0.79 ms | 105.43 ms | 1099.06 ms | 0.22272 |
| **512x512** | 4083.99 ms | 1.82 ms | 493.06 ms | 4578.86 ms | 0.24108 |
