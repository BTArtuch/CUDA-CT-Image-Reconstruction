#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

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
    std::string out_file = "results/recon_cpp_" + std::to_string(size) + ".bin";

    std::vector<float> filtered_sino(num_angles * num_detectors);
    std::vector<float> angles(num_angles);
    std::vector<float> recon(size * size, 0.0f);

    if (!read_binary(sino_file, filtered_sino) || !read_binary(angles_file, angles)) {
        std::cerr << "Error: Could not read input binaries. Did you run the Python script?\n";
        return 1;
    }

    float center = (size - 1) / 2.0f;
    float det_center = (num_detectors - 1) / 2.0f;
    float d_theta = M_PI / num_angles;

    std::cout << "--- Running C++ FBP Benchmark: " << size << "x" << size << " ---" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Core Backprojection Loop
    for (int i = 0; i < num_angles; ++i) {
        float theta = angles[i];
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);
        const float* proj_slice = &filtered_sino[i * num_detectors];

        for (int y = 0; y < size; ++y) {
            float Y = center - y; // Matches Python's -y meshgrid
            
            for (int x = 0; x < size; ++x) {
                float X = x - center;
                
                // Calculate detector intersection
                float t = X * cos_t + Y * sin_t + det_center;
                
                int t0 = static_cast<int>(std::floor(t));
                int t1 = t0 + 1;
                
                if (t0 >= 0 && t1 < num_detectors) {
                    float wt1 = t - t0;
                    float wt0 = 1.0f - wt1;
                    
                    recon[y * size + x] += (proj_slice[t0] * wt0 + proj_slice[t1] * wt1);
                }
            }
        }
    }

    // Apply angular differential
    for (int i = 0; i < size * size; ++i) {
        recon[i] *= d_theta;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    std::cout << "C++ Backprojection: " << duration.count() << " ms\n\n";

    write_binary(out_file, recon);
    return 0;
}