# FlyBrain Windows Environment Acceptance

## Host
- Windows version: Windows 10 Pro, build 26200
- WSL version: 2.7.13.0; kernel 6.18.33.2-microsoft-standard-WSL2
- Ubuntu version: Ubuntu 24.04.5 LTS
- Architecture: x86_64
- RAM: 64 GiB host; about 30 GiB available inside WSL during setup
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB
- NVIDIA driver: Windows KMD 610.88; WSL NVIDIA-SMI 610.57.01

## Python / ML
- Python: 3.12.3
- PyTorch: 2.14.0+cu130
- PyTorch CUDA: 13.0
- CUDA available: true
- GPU compute capability: 12.0
- dense CUDA smoke: PASS
- sparse CUDA smoke: PASS

## Project
- project path: `/home/denny/projects/flybrain_lab_4spark_v0_1`
- virtual environment: `/home/denny/projects/flybrain-env`
- unit tests: 13 passed
- modified files: reports and generated data only

## MaleCNS
- `brain.npz` checksum: PASS
- `weights.npz` checksum: PASS
- converted graph: `data/male-v1.npz`
- N: 166700
- nnz: 25582938
- graph size: 93 MiB (CSR sparse)
- graph validation: finite weights, in-range indices, shape 166700 x 166700

## RTX 5070 Ti Benchmark
### batch=1
- env steps/s: 304.8295
- mean spike fraction: 0.0003125
- input contrast: 0.0005856
- CUDA peak allocated: 313990656 bytes
- result: PASS

### batch=4
- env steps/s: 298.3406 aggregate
- mean spike fraction: 0.0003140
- input contrast: 0.0006276
- CUDA peak allocated: 335787520 bytes
- result: PASS

## Problems / Changes
- Host is Windows 10 Pro rather than the plan's Windows 11 target.
- GPU is RTX 5070 Ti rather than RTX 5070; it is visible in Windows and WSL and reports compute capability 12.0.
- WSL reported about 30 GiB available memory during setup, below the plan's suggested 40 GiB; disk space was ample.
- The complete delivery archive contained the P0 project package and P1 plan. P1 was not executed.
- MaleCNS downloads initially timed out; after the host VPN was switched to global TUN mode, the project downloader completed both downloads and checksum verification.
- No NVIDIA Linux kernel driver was installed in WSL.

## Final Gate
P0_ENVIRONMENT_READY = YES
