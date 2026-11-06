# University GRID P40-8Q: initial setup

Verified from the supplied screenshots on 6 November 2026:

| Resource | Reported value |
| --- | --- |
| Guest GPU | GRID P40-8Q |
| GPU allocation | 8,192 MiB total; 6,433 MiB free at query time |
| NVIDIA driver | 535.161.08 |
| `nvidia-smi` CUDA capability header | 12.2 |
| OS | Ubuntu 24.04.4 LTS |
| Logical CPUs | 8 |
| System RAM | 31 GiB total; about 28 GiB available |
| Disk | About 170 GiB available on `/` |

The full physical P40 can have 24 GB, but this VM receives 8 GiB. Keep that distinction in batch/resolution planning. Actual usable memory changes with the desktop and other workloads.

Use Python 3.12 and PyTorch 2.7.1 CUDA 11.8, matching the project's working CUDA/PyTorch release. The driver is expected to support CUDA 11.8 applications through backward compatibility. The `12.2` driver header does not require a `cu122` wheel or a local CUDA-toolkit installation for this binary-wheel test. Leave the existing university vGPU driver in place. [Official PyTorch commands](https://pytorch.org/get-started/previous-versions/), [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Run on the university Ubuntu machine. These steps have not been executed remotely by the assistant.

## 1. Create an isolated environment

Ubuntu 24.04 provides the Python 3.12 venv package. If it is already installed, apt leaves it installed. If this account has no sudo privileges, ask the university administrator for that package and continue with the user-owned environment. [Ubuntu package information](https://packages.ubuntu.com/noble/python3.12-venv).

```bash
sudo apt update
sudo apt install -y python3.12-venv

python3.12 -m venv ~/paper-p40-venv
source ~/paper-p40-venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

This initial check needs PyTorch only; install the remaining project dependencies after the GPU test and source transfer. No model/dataset training is started here.

## 2. Exercise CUDA training operations

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("Wheel CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is unavailable; keep the complete output/error."

device = torch.cuda.get_device_properties(0)
print("GPU:", device.name)
print("Compute capability:", torch.cuda.get_device_capability(0))
print("GPU memory GiB:", round(device.total_memory / 2**30, 2))
print("Wheel architectures:", torch.cuda.get_arch_list())

model = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randn(1, 3, 128, 128, device="cuda")
loss = model(x).square().mean()
loss.backward()
optimizer.step()
torch.cuda.synchronize()
assert torch.isfinite(loss).item(), "Non-finite test loss"
print("GPU forward/backward/optimizer: PASS")
PY
```

Send the output, including any exception. `PASS` establishes that these CUDA/cuDNN/optimizer operations work in the VM; it does not establish that the full paper model fits at 768 or that segmentation quality has improved. That requires the project's full training-step profile and subsequent data-driven training.

## 3. What follows

Transfer the local foundation source and approved checkpoint/data, install the project's pinned training dependencies, then profile the real model at FP32, physical batch 1, 768-pixel input. Preserve memory headroom and use gradient accumulation for larger effective batches. Adjust views or memory-saving implementation if a single full step does not fit. Do not run the historical v3 training script unchanged or expect an unpublished local branch to exist on GitHub.
