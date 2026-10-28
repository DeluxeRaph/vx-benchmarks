# Vx versus CUDA: vector addition on Jetson Orin Nano

Measured October 28, 2026. CUDA was faster in this vector-add microbenchmark.
At 1,048,576 elements, Vx's CUDA-event interval was about 4.1% longer; its
synchronized host loop took about 15.6% longer. This does not establish a
general speed ranking for the languages.

## Results

Medians across five process runs. GPU interval: median of each run's 501 samples,
then median of those five medians. Host: mean per iteration in each process,
then median across five runs. Smaller is better.

| Elements | Vx event interval (us) | CUDA event interval (us) | Vx host + GPU + wait (us) | CUDA host + GPU + wait (us) |
|---:|---:|---:|---:|---:|
| 4 | 17.00 | 14.66 | 49.58 | 38.88 |
| 1,024 | 18.00 | 16.13 | 53.44 | 41.04 |
| 65,536 | 31.00 | 26.62 | 72.28 | 68.03 |# Vx versus CUDA: vector addition on Jetson Orin Nano

Measured October 28, 2026. CUDA was faster in this vector-add microbenchmark.
At 1,048,576 elements, Vx's CUDA-event interval was about 4.1% longer; its
synchronized host loop took about 15.6% longer. This does not establish a
general speed ranking for the languages.

## Results

Medians across five process runs. GPU interval: median of each run's 501 samples,
then median of those five medians. Host: mean per iteration in each process,
then median across five runs. Smaller is better.

| Elements | Vx event interval (us) | CUDA event interval (us) | Vx host + GPU + wait (us) | CUDA host + GPU + wait (us) |
|---:|---:|---:|---:|---:|
| 4 | 17.00 | 14.66 | 49.58 | 38.88 |
| 1,024 | 18.00 | 16.13 | 53.44 | 41.04 |
| 65,536 | 31.00 | 26.62 | 72.28 | 68.03 |
| 1,048,576 | 226.00 | 217.06 | 289.19 | 250.25 |

## Transfers and allocation

Measured separately in the non-instrumented processes; includes allocation.
Upload stages A, B, and the initially zero result to GPU allocations, matching
the hello-world program. Download allocates a new CPU output and copies back.
These are five-run medians of one transfer stage per process, not a bandwidth benchmark.

| Elements | Vx upload (ms) | CUDA upload (ms) | Vx download (ms) | CUDA download (ms) |
|---:|---:|---:|---:|---:|
| 4 | 0.972 | 0.968 | 0.329 | 0.152 |
| 1,024 | 0.995 | 1.011 | 0.306 | 0.118 |
| 65,536 | 2.493 | 2.327 | 0.898 | 0.727 |
| 1,048,576 | 7.169 | 5.854 | 5.139 | 4.934 |

## Method

- Same float32 addition and input values: A[i] = i % 97, B[i] = i % 31. Every output element checked on every process run.
- Same launch shape: 1 block x 4 threads; then 4, 256, or 4096 blocks x 256 threads.
- Vx 0.0.3 at commit 93cdd0a24966b48cf9dd90e7347fb2e4e5dde2ab; Vx program compiled with -O3 to an object and linked into a native executable with the Vx CUDA runtime.
- CUDA kernel compiled by the installed NVRTC for compute_87; native C++ host compiled with Clang 22.1.8 -O3. CUDA source is cuda_add.cu. No special math approximation is needed for addition.
- 2,000 warm-up launches, followed by 501 measured launches; five process runs for each size and mode. Alternate which implementation runs first.
- Timing mode: CUDA events bracket each launch and synchronize on its ending event. Vx uses the upstream runtime's VX_TIME_KERNEL instrumentation; CUDA uses the same event mechanism. A timing line for every Vx launch confirms GPU image execution; missing timing samples fail the harness.
- Host mode: event instrumentation disabled. Measure 501 launches and an explicit cudaDeviceSynchronize after each, with buffers already resident. Includes runtime dispatch, GPU work, and waiting; excludes transfers and correctness checking.
- Vx uses its monotonic std::time clock; CUDA uses std::chrono::steady_clock.
- Compile/link, NVRTC compilation, CUDA context initialization, and first module loading are excluded from the reported operation timings. The stored process_ms metric includes startup (including CUDA NVRTC compilation) and MUST NOT be used as a fair Vx/CUDA runtime comparison.
- Python only orchestrates builds and process runs, and is outside all timed operation regions. Both applications are native CPU executables launching GPU kernels.
- Jetson Orin Nano in its existing 15 W mode; dynamic CPU/GPU/memory clocks were left unchanged. System was not isolated or clock-locked.

## Limits

CUDA-event intervals are not a profiler measurement of arithmetic instructions alone.
They can include GPU-stream idle gaps while the CPU submits work. This matters
especially for four elements, where launch overhead dwarfs the calculation.
Vx's existing timing log rounds milliseconds to three decimals (1 us resolution).
The small timing differences should therefore not be read as precise hardware limits.

The longer warm-up stabilized the large-array event measurements. Across the five
runs, Vx's large-array medians ranged from 224 to 226 us; CUDA's ranged from
215.97 to 217.50 us. Some synchronized host runs still had scheduling outliers;
the summary uses five-run medians and retains every run in results.json.

The preliminary 20-warm-up run had substantial clock-related variability and was
superseded by the 2,000-warm-up results. Its raw results remain on the Jetson as
preliminary-results.json and preliminary-summary.json.

Vector addition is a simple memory/launch benchmark. It says nothing by itself
about matrix multiplication, camera filters, or application-wide performance.

## Reproduce

After configuring the build environment in [SETUP.md](../../SETUP.md):

```sh
cd benchmarks/vector-addition
python3 -u bench.py
```

The harness rebuilds native binaries, raises its own stack limit to 64 MiB for
Vx's fixed-size host tensors, checks correctness, records per-run logs, and writes
results.json and summary.json. See ../../SETUP.md for build environment requirements.
No system packages, power settings, or camera code were changed for the benchmark.

| 1,048,576 | 226.00 | 217.06 | 289.19 | 250.25 |

## Transfers and allocation

Measured separately in the non-instrumented processes; includes allocation.
Upload stages A, B, and the initially zero result to GPU allocations, matching
the hello-world program. Download allocates a new CPU output and copies back.
These are five-run medians of one transfer stage per process, not a bandwidth benchmark.

| Elements | Vx upload (ms) | CUDA upload (ms) | Vx download (ms) | CUDA download (ms) |
|---:|---:|---:|---:|---:|
| 4 | 0.972 | 0.968 | 0.329 | 0.152 |
| 1,024 | 0.995 | 1.011 | 0.306 | 0.118 |
| 65,536 | 2.493 | 2.327 | 0.898 | 0.727 |
| 1,048,576 | 7.169 | 5.854 | 5.139 | 4.934 |

## Method

- Same float32 addition and input values: A[i] = i % 97, B[i] = i % 31. Every output element checked on every process run.
- Same launch shape: 1 block x 4 threads; then 4, 256, or 4096 blocks x 256 threads.
- Vx 0.0.3 at commit 93cdd0a24966b48cf9dd90e7347fb2e4e5dde2ab; Vx program compiled with -O3 to an object and linked into a native executable with the Vx CUDA runtime.
- CUDA kernel compiled by the installed NVRTC for compute_87; native C++ host compiled with Clang 22.1.8 -O3. CUDA source is cuda_add.cu. No special math approximation is needed for addition.
- 2,000 warm-up launches, followed by 501 measured launches; five process runs for each size and mode. Alternate which implementation runs first.
- Timing mode: CUDA events bracket each launch and synchronize on its ending event. Vx uses the upstream runtime's VX_TIME_KERNEL instrumentation; CUDA uses the same event mechanism. A timing line for every Vx launch confirms GPU image execution; missing timing samples fail the harness.
- Host mode: event instrumentation disabled. Measure 501 launches and an explicit cudaDeviceSynchronize after each, with buffers already resident. Includes runtime dispatch, GPU work, and waiting; excludes transfers and correctness checking.
- Vx uses its monotonic std::time clock; CUDA uses std::chrono::steady_clock.
- Compile/link, NVRTC compilation, CUDA context initialization, and first module loading are excluded from the reported operation timings. The stored process_ms metric includes startup (including CUDA NVRTC compilation) and MUST NOT be used as a fair Vx/CUDA runtime comparison.
- Python only orchestrates builds and process runs, and is outside all timed operation regions. Both applications are native CPU executables launching GPU kernels.
- Jetson Orin Nano in its existing 15 W mode; dynamic CPU/GPU/memory clocks were left unchanged. System was not isolated or clock-locked.

## Limits

CUDA-event intervals are not a profiler measurement of arithmetic instructions alone.
They can include GPU-stream idle gaps while the CPU submits work. This matters
especially for four elements, where launch overhead dwarfs the calculation.
Vx's existing timing log rounds milliseconds to three decimals (1 us resolution).
The small timing differences should therefore not be read as precise hardware limits.

The longer warm-up stabilized the large-array event measurements. Across the five
runs, Vx's large-array medians ranged from 224 to 226 us; CUDA's ranged from
215.97 to 217.50 us. Some synchronized host runs still had scheduling outliers;
the summary uses five-run medians and retains every run in results.json.

The preliminary 20-warm-up run had substantial clock-related variability and was
superseded by the 2,000-warm-up results. Its raw results remain on the Jetson as
preliminary-results.json and preliminary-summary.json.

Vector addition is a simple memory/launch benchmark. It says nothing by itself
about matrix multiplication, camera filters, or application-wide performance.

## Reproduce

After configuring the build environment in [SETUP.md](../SETUP.md):

```sh
cd benchmark
python3 -u bench.py
```

The harness rebuilds native binaries, raises its own stack limit to 64 MiB for
Vx's fixed-size host tensors, checks correctness, records per-run logs, and writes
results.json and summary.json. See ../SETUP.md for build environment requirements.
No system packages, power settings, or camera code were changed for the benchmark.
