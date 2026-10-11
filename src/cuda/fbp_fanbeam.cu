#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to read binary files
bool read_binary(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return true;
}

// Helper to write binary files
bool write_binary(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return true;
}

// CUDA Kernel: Fan-Beam Distance-Weighted Backprojection
__global__ void backproject_fanbeam_kernel(
    float* recon, const float* sino, const float* betas,
    int size, int num_angles, int num_detectors,
    float center, float D, float gamma_min, float d_gamma, float d_beta) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size) return;

    float X = x - center;
    float Y = center - y;
    float pixel_value = 0.0f;

    for (int i = 0; i < num_angles; ++i) {
        float beta = betas[i];
        
        float cos_b = __cosf(beta);
        float sin_b = __sinf(beta);
        
        // 1. Calculate rotated spatial coordinates relative to the iso-ray
        float s = X * cos_b + Y * sin_b;
        float t = -X * sin_b + Y * cos_b;
        
        // 2. Calculate the distance squared from the point source to the pixel (L^2)
        float L2 = s * s + (D + t) * (D + t);
        
        // 3. Find the exact divergent detector angle (gamma) for this pixel
        float gamma = atan2f(s, D + t);
        
        // 4. Map the continuous angle to a discrete detector bin
        float gamma_bin = (gamma - gamma_min) / d_gamma;
        
        int g0 = floorf(gamma_bin);
        int g1 = g0 + 1;
        
        if (g0 >= 0 && g1 < num_detectors) {
            float wg1 = gamma_bin - g0;
            float wg0 = 1.0f - wg1;
            
            int slice_offset = i * num_detectors;
            float interp_val = (sino[slice_offset + g0] * wg0) + (sino[slice_offset + g1] * wg1);
            
            // 5. The Inverse-Square Distance Accumulation
            pixel_value += interp_val / L2;
        }
    }

    // Final scaling by d_beta
    recon[y * size + x] = pixel_value * d_beta;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_size> <num_angles>\n";
        return 1;
    }

    int size = std::stoi(argv[1]);
    int num_angles = std::stoi(argv[2]);
    int num_detectors = size;

    std::string sino_file = "data/fan_filtered_sino_" + std::to_string(size) + "_" + std::to_string(num_angles) + ".bin";
    std::string betas_file = "data/fan_betas_" + std::to_string(num_angles) + ".bin";
    std::string gammas_file = "data/fan_gammas_" + std::to_string(size) + ".bin";
    std::string out_file = "results/recon_cuda_fanbeam_" + std::to_string(size) + ".bin";

    std::vector<float> h_sino(num_angles * num_detectors);
    std::vector<float> h_betas(num_angles);
    std::vector<float> h_gammas(num_detectors);
    std::vector<float> h_recon(size * size, 0.0f);

    if (!read_binary(sino_file, h_sino) || !read_binary(betas_file, h_betas) || !read_binary(gammas_file, h_gammas)) {
        std::cerr << "Error reading binaries. Did you run the Python generator?\n";
        return 1;
    }

    // Calculate geometry bounds directly from the generated gamma array
    float D = size * 1.5f; 
    float gamma_min = h_gammas[0];
    float d_gamma = h_gammas[1] - h_gammas[0];
    float d_beta = (2.0f * M_PI) / num_angles;
    float center = (size - 1) / 2.0f;

    // Allocate Device Memory
    float *d_sino, *d_betas, *d_recon;
    cudaMalloc(&d_sino, h_sino.size() * sizeof(float));
    cudaMalloc(&d_betas, h_betas.size() * sizeof(float));
    cudaMalloc(&d_recon, h_recon.size() * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_sino, h_sino.data(), h_sino.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_betas, h_betas.data(), h_betas.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((size + threads.x - 1) / threads.x, (size + threads.y - 1) / threads.y);

    std::cout << "--- Running CUDA Fan-Beam Benchmark: " << size << "x" << size << " ---" << std::endl;

    cudaDeviceSynchronize();
    auto start_time = std::chrono::high_resolution_clock::now();

    backproject_fanbeam_kernel<<<blocks, threads>>>(
        d_recon, d_sino, d_betas, 
        size, num_angles, num_detectors, 
        center, D, gamma_min, d_gamma, d_beta
    );

    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    std::cout << "CUDA Fan-Beam Backprojection: " << duration.count() << " ms\n\n";

    // Retrieve result and cleanup
    cudaMemcpy(h_recon.data(), d_recon, h_recon.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sino);
    cudaFree(d_betas);
    cudaFree(d_recon);

    write_binary(out_file, h_recon);
    return 0;
}