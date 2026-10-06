import os
import numpy as np
import matplotlib.pyplot as plt

def generate_benchmark_plot():
    # Hardcoded execution times from your terminal output
    resolutions = ['128x128', '256x256', '512x512']
    
    python_times = [28.39, 105.43, 493.06]
    cpp_times = [9.02, 21.49, 72.73]
    cuda_times = [0.21, 0.58, 2.03]

    x = np.arange(len(resolutions))
    width = 0.25  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the grouped bars
    rects1 = ax.bar(x - width, python_times, width, label='Python', color='#4B8BBE')
    rects2 = ax.bar(x, cpp_times, width, label='C++', color='#f34b7d')
    rects3 = ax.bar(x + width, cuda_times, width, label='CUDA', color='#76b900')

    # Formatting
    ax.set_ylabel('Execution Time (ms) - Log Scale', fontsize=12)
    ax.set_title('FBP Backprojection Performance: Python vs C++ vs CUDA', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions, fontsize=12)
    ax.legend(fontsize=12)
    
    # Use a log scale so the 0.2ms CUDA bar isn't invisible next to the 500ms Python bar
    ax.set_yscale('log')
    
    # Add a grid for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Helper function to attach text labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/benchmark_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Benchmark plot saved to: {out_path}")

if __name__ == "__main__":
    generate_benchmark_plot()