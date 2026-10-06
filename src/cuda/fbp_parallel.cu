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

// CUDA Kernel: Each thread calculates the final value for one pixel (x, y)
__global__ void backproject_kernel(
    float* recon, const float* sino, const float* angles,
    int size, int num_angles, int num_detectors,
    float center, float det_center, float d_theta) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size) return;

    float X = x - center;
    float Y = center - y;
    float pixel_value = 0.0f;

    for (int i = 0; i < num_angles; ++i) {
        float theta = angles[i];
        
        // __cosf and __sinf are fast hardware intrinsics
        float cos_t = __cosf(theta);
        float sin_t = __sinf(theta);
        
        float t = X * cos_t + Y * sin_t + det_center;
        
        int t0 = floorf(t);
        int t1 = t0 + 1;
        
        if (t0 >= 0 && t1 < num_detectors) {
            float wt1 = t - t0;
            float wt0 = 1.0f - wt1;
            
            int slice_offset = i * num_detectors;
            pixel_value += (sino[slice_offset + t0] * wt0 + sino[slice_offset + t1] * wt1);
        }
    }

    recon[y * size + x] = pixel_value * d_theta;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_size> <num_angles>\n";
        return 1;
    }

    int size = std::stoi(argv[1]);
    int num_angles = std::stoi(argv[2]);
    int num_detectors = size;

    std::string sino_file = "data/filtered_sino_" + std::to_string(size) + "_" + std::to_string(num_angles) + ".bin";
    std::string angles_file = "data/angles_" + std::to_string(num_angles) + ".bin";
    std::string out_file = "results/recon_cuda_" + std::to_string(size) + ".bin";

    std::vector<float> h_sino(num_angles * num_detectors);
    std::vector<float> h_angles(num_angles);
    std::vector<float> h_recon(size * size, 0.0f);

    if (!read_binary(sino_file, h_sino) || !read_binary(angles_file, h_angles)) {
        std::cerr << "Error reading binaries.\n";
        return 1;
    }

    // Allocate Device Memory
    float *d_sino, *d_angles, *d_recon;
    size_t sino_bytes = h_sino.size() * sizeof(float);
    size_t angles_bytes = h_angles.size() * sizeof(float);
    size_t recon_bytes = h_recon.size() * sizeof(float);

    cudaMalloc(&d_sino, sino_bytes);
    cudaMalloc(&d_angles, angles_bytes);
    cudaMalloc(&d_recon, recon_bytes);

    // Copy data to device
    cudaMemcpy(d_sino, h_sino.data(), sino_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_angles, h_angles.data(), angles_bytes, cudaMemcpyHostToDevice);

    float center = (size - 1) / 2.0f;
    float det_center = (num_detectors - 1) / 2.0f;
    float d_theta = M_PI / num_angles;

    dim3 threads(16, 16);
    dim3 blocks((size + threads.x - 1) / threads.x, (size + threads.y - 1) / threads.y);

    std::cout << "--- Running CUDA FBP Benchmark: " << size << "x" << size << " ---" << std::endl;

    // Start timing (measuring purely kernel execution time to match C++ backproject loop)
    cudaDeviceSynchronize();
    auto start_time = std::chrono::high_resolution_clock::now();

    backproject_kernel<<<blocks, threads>>>(
        d_recon, d_sino, d_angles, 
        size, num_angles, num_detectors, 
        center, det_center, d_theta
    );

    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    std::cout << "CUDA Backprojection: " << duration.count() << " ms\n\n";

    // Retrieve result and cleanup
    cudaMemcpy(h_recon.data(), d_recon, recon_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_sino);
    cudaFree(d_angles);
    cudaFree(d_recon);

    write_binary(out_file, h_recon);
    return 0;
}