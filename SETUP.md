# Running the benchmarks

Use Linux ARM64 on a Jetson Orin Nano with a working CUDA driver.

1. Build [Vx](https://github.com/vx-lang/Vx) with LLVM/MLIR 22 and a CUDA toolkit, including its CUDA dispatch runtime and `vx_std_core`. The recorded tests use Vx commit `93cdd0a24966b48cf9dd90e7347fb2e4e5dde2ab`.
2. Put Clang 22 and Python 3 on `PATH`, and configure your library paths for the Vx and CUDA runtimes.
3. Set these paths for your installation:

```sh
export VXC_ROOT=/path/to/Vx
export MLIR_SYS_220_PREFIX=/path/to/llvm-22
export CUDA_HOME=/path/to/cuda
# Optional: override the CUDA driver library directory on Jetson.
export CUDA_DRIVER_LIB_DIR=/opt/nvidia/l4t-gpu-libs/nvgpu
cd benchmark
python3 -u bench.py
```

The harness expects a debug-profile Vx build at `$VXC_ROOT/target/debug/vxc`, its `libvx_std_core.so`, and CUDA dispatch/shim libraries in the Cargo build output. Set `VX_DISPATCH_LIB` to select a particular dispatch library if multiple builds exist.

The CUDA toolkit must have headers and linkable `libcuda`, `libcudart`, `libcublas`, and `libnvrtc` libraries. An unversioned `libnvrtc.so` development symlink must already exist. If CUDA components come from Python wheels, extra CUDA CRT and CCCL headers may be needed; the wheel alone may not be a complete development toolkit. This repository does not install dependencies or change system settings.

The harness compiles both native executables, verifies every output, and writes `results.json`, `summary.json`, and per-run logs in `benchmark/`. A run replaces the saved result JSON files, so copy them first if you want to retain the recorded results.

Python orchestrates the test; it is outside all timed operation regions. Vx’s native program uses the Vx CUDA runtime. Both implementations use identical inputs and GPU launch shapes.
