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