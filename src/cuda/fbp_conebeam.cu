#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool write_binary(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return true;
}

// 1. Generate 3D Mathematical Phantom
__global__ void generate_3d_phantom(float* phantom, int size, float center) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= size || y >= size || z >= size) return;

    float X = x - center;
    float Y = center - y;
    float Z = z - center;
    
    // Geometry: A cylinder torso with a dense inner sphere and a small offset sphere
    float r_cyl = sqrtf(X*X + Y*Y);
    float r_sph = sqrtf(X*X + Y*Y + Z*Z);
    float r_small = sqrtf((X - size*0.2f)*(X - size*0.2f) + Y*Y + Z*Z);
    
    float val = 0.0f;
    if (r_cyl < size * 0.4f && abs(Z) < size * 0.4f) val = 0.3f; 
    if (r_sph < size * 0.15f) val += 0.5f; 
    if (r_small < size * 0.1f) val += 0.4f; 
    
    phantom[z * size * size + y * size + x] = val;
}

// 2. 3D Cone-Beam Forward Projection (Ray Casting)
__global__ void forward_project_conebeam(
    float* sino, const float* phantom, const float* betas,
    int size, int num_angles, int num_u, int num_v,
    float center, float D, float d_u, float d_v)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int a = blockIdx.z * blockDim.z + threadIdx.z;

    if (u >= num_u || v >= num_v || a >= num_angles) return;

    float beta = betas[a];
    float cos_b = __cosf(beta);
    float sin_b = __sinf(beta);

    float u_coord = (u - (num_u - 1) / 2.0f) * d_u;
    float v_coord = (v - (num_v - 1) / 2.0f) * d_v;

    // Normalize directional vector from source to detector
    float ray_len = sqrtf(u_coord * u_coord + D * D + v_coord * v_coord);
    float dx_rot = u_coord / ray_len;
    float dy_rot = D / ray_len;
    float dz_rot = v_coord / ray_len;

    // Rotate to world frame
    float dir_X = dx_rot * cos_b - dy_rot * sin_b;
    float dir_Y = dx_rot * sin_b + dy_rot * cos_b;
    float dir_Z = dz_rot;

    // Source position in world frame
    float src_X = D * sin_b;
    float src_Y = -D * cos_b;
    float src_Z = 0.0f;

    float sum = 0.0f;
    float step_size = 0.5f; 
    int num_steps = (D * 2.5f) / step_size; // Ensure ray passes entirely through volume

    for (int i = 0; i < num_steps; ++i) {
        float px = src_X + dir_X * (i * step_size);
        float py = src_Y + dir_Y * (i * step_size);
        float pz = src_Z + dir_Z * (i * step_size);

        int vx = roundf(px + center);
        int vy = roundf(center - py);
        int vz = roundf(pz + center);

        if (vx >= 0 && vx < size && vy >= 0 && vy < size && vz >= 0 && vz < size) {
            sum += phantom[vz * size * size + vy * size + vx] * step_size;
        }
    }
    sino[a * num_u * num_v + v * num_u + u] = sum;
}

// 3. FDK Pre-Weighting & Spatial Ram-Lak Filter
__global__ void filter_conebeam(
    float* filtered_sino, const float* sino, 
    int num_angles, int num_u, int num_v, float D, float d_u, float d_v)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int a = blockIdx.z * blockDim.z + threadIdx.z;

    if (u >= num_u || v >= num_v || a >= num_angles) return;

    float v_coord = (v - (num_v - 1) / 2.0f) * d_v;
    float center_u = (num_u - 1) / 2.0f;
    float sum = 0.0f;
    
    for (int k = 0; k < num_u; ++k) {
        int offset = u - k;
        float filter_val = 0.0f;
        
        if (offset == 0) {
            filter_val = 0.25f / (d_u * d_u);
        } else if (offset % 2 != 0) {
            float dist = offset * d_u;
            filter_val = -1.0f / (M_PI * M_PI * dist * dist);
        }
        
        if (filter_val != 0.0f) {
            float k_u_coord = (k - center_u) * d_u;
            // Apply FDK Divergent Pre-Weighting dynamically during convolution
            float weight = D / sqrtf(D * D + k_u_coord * k_u_coord + v_coord * v_coord);
            float val = sino[a * num_u * num_v + v * num_u + k] * weight;
            sum += val * filter_val;
        }
    }
    filtered_sino[a * num_u * num_v + v * num_u + u] = sum * d_u; 
}

// 4. 3D FDK Distance-Weighted Backprojection
__global__ void backproject_conebeam_kernel(
    float* recon, const float* sino, const float* betas,
    int size, int num_angles, int num_u, int num_v,
    float center, float center_z, float D, float d_u, float d_v, float d_beta) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= size || y >= size || z >= size) return;

    float X = x - center;
    float Y = center - y;
    float Z = z - center_z;
    float voxel_value = 0.0f;

    for (int i = 0; i < num_angles; ++i) {
        float beta = betas[i];
        float cos_b = __cosf(beta);
        float sin_b = __sinf(beta);
        
        float s = X * cos_b + Y * sin_b;
        float t = -X * sin_b + Y * cos_b;
        
        float U = D / (D + t);
        float u_coord = U * s;
        float v_coord = U * Z;
        
        float u_bin = u_coord / d_u + (num_u - 1) / 2.0f;
        float v_bin = v_coord / d_v + (num_v - 1) / 2.0f;
        
        int u0 = floorf(u_bin);
        int u1 = u0 + 1;
        int v0 = floorf(v_bin);
        int v1 = v0 + 1;
        
        if (u0 >= 0 && u1 < num_u && v0 >= 0 && v1 < num_v) {
            float wu1 = u_bin - u0;
            float wu0 = 1.0f - wu1;
            float wv1 = v_bin - v0;
            float wv0 = 1.0f - wv1;
            
            int slice_offset = i * num_u * num_v;
            
            float val_v0 = (sino[slice_offset + v0 * num_u + u0] * wu0) + 
                           (sino[slice_offset + v0 * num_u + u1] * wu1);
            float val_v1 = (sino[slice_offset + v1 * num_u + u0] * wu0) + 
                           (sino[slice_offset + v1 * num_u + u1] * wu1);
            
            float interp_val = (val_v0 * wv0) + (val_v1 * wv1);
            voxel_value += interp_val * (U * U);
        }
    }
    // Multiply by 0.5f because a 360-degree rotation captures every ray path twice
    recon[z * (size * size) + y * size + x] = voxel_value * d_beta * 0.5f;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <volume_size> <num_angles>\n";
        return 1;
    }

    int size = std::stoi(argv[1]);
    int num_angles = std::stoi(argv[2]);
    int num_u = size;
    int num_v = size;

    float D = size * 1.5f; 
    float d_u = 1.0f;
    float d_v = 1.0f;
    float d_beta = (2.0f * M_PI) / num_angles;
    float center = (size - 1) / 2.0f;
    float center_z = (size - 1) / 2.0f;

    std::vector<float> h_betas(num_angles);
    for (int i = 0; i < num_angles; ++i) h_betas[i] = i * d_beta;

    float *d_betas;
    cudaMalloc(&d_betas, num_angles * sizeof(float));
    cudaMemcpy(d_betas, h_betas.data(), num_angles * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "\n--- Running End-to-End 3D Cone-Beam Pipeline: " << size << "^3 ---" << std::endl;

    // ==========================================
    // Phase 1: Phantom Generation & Export
    // ==========================================
    float *d_phantom;
    cudaMalloc(&d_phantom, size * size * size * sizeof(float));
    
    dim3 threads3D(8, 8, 8);
    dim3 blocks3D((size + 7) / 8, (size + 7) / 8, (size + 7) / 8);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    generate_3d_phantom<<<blocks3D, threads3D>>>(d_phantom, size, center);
    cudaDeviceSynchronize();

    std::vector<float> h_slice(size * size);
    int center_idx = size / 2;
    int edge_idx = static_cast<int>(size * 0.85f); // High slice to reveal Cone-Beam artifact

    cudaMemcpy(h_slice.data(), d_phantom + center_idx*size*size, size*size*sizeof(float), cudaMemcpyDeviceToHost);
    write_binary("data/phantom_cone_center_" + std::to_string(size) + ".bin", h_slice);
    
    cudaMemcpy(h_slice.data(), d_phantom + edge_idx*size*size, size*size*sizeof(float), cudaMemcpyDeviceToHost);
    write_binary("data/phantom_cone_edge_" + std::to_string(size) + ".bin", h_slice);

    // ==========================================
    // Phase 2: Forward Projection
    // ==========================================
    float *d_sino;
    cudaMalloc(&d_sino, num_angles * num_u * num_v * sizeof(float));
    
    dim3 threads2D(8, 8, 8);
    dim3 blocks2D((num_u + 7) / 8, (num_v + 7) / 8, (num_angles + 7) / 8);

    forward_project_conebeam<<<blocks2D, threads2D>>>(d_sino, d_phantom, d_betas, size, num_angles, num_u, num_v, center, D, d_u, d_v);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "3D Forward Projection: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // Free phantom to keep VRAM usage perfectly low
    cudaFree(d_phantom);

    // ==========================================
    // Phase 3: Spatial FDK Filter
    // ==========================================
    float *d_filtered;
    cudaMalloc(&d_filtered, num_angles * num_u * num_v * sizeof(float));

    filter_conebeam<<<blocks2D, threads2D>>>(d_filtered, d_sino, num_angles, num_u, num_v, D, d_u, d_v);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "3D FDK Ramp Filtering: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

    cudaFree(d_sino);

    // ==========================================
    // Phase 4: FDK Backprojection & Export
    // ==========================================
    float *d_recon;
    cudaMalloc(&d_recon, size * size * size * sizeof(float));

    backproject_conebeam_kernel<<<blocks3D, threads3D>>>(d_recon, d_filtered, d_betas, size, num_angles, num_u, num_v, center, center_z, D, d_u, d_v, d_beta);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "3D FDK Backprojection: " << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";
    std::cout << "Total Pipeline Time:   " << std::chrono::duration<double, std::milli>(t3 - t0).count() << " ms\n\n";

    cudaMemcpy(h_slice.data(), d_recon + center_idx*size*size, size*size*sizeof(float), cudaMemcpyDeviceToHost);
    write_binary("results/recon_cone_center_" + std::to_string(size) + ".bin", h_slice);

    cudaMemcpy(h_slice.data(), d_recon + edge_idx*size*size, size*size*sizeof(float), cudaMemcpyDeviceToHost);
    write_binary("results/recon_cone_edge_" + std::to_string(size) + ".bin", h_slice);

    cudaFree(d_filtered);
    cudaFree(d_recon);
    cudaFree(d_betas);

    return 0;
}