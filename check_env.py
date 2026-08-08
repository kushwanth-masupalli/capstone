"""
Phase 0 environment checker — run this BEFORE writing any model code,
and have every team member run it individually since you're on local
GPU machines with potentially different drivers.

Usage:
    python check_env.py
"""
import subprocess
import sys


def check_nvidia_driver():
    print("=" * 60)
    print("1. Checking NVIDIA driver (nvidia-smi)")
    print("=" * 60)
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            print(out.stdout)
            return True
        else:
            print("nvidia-smi failed to run. Is a GPU + driver installed?")
            print(out.stderr)
            return False
    except FileNotFoundError:
        print("nvidia-smi not found. No NVIDIA driver detected on this machine.")
        print("-> If you believe you have a GPU, install/update NVIDIA drivers first.")
        return False


def check_torch():
    print("=" * 60)
    print("2. Checking PyTorch + CUDA")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("torch is not installed yet.")
        print_install_hint()
        return

    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    if not cuda_available:
        print("\n*** CUDA NOT AVAILABLE. Training will silently run on CPU ***")
        print("*** and be 20-50x slower. Fix this before training anything. ***\n")
        print_install_hint()
        return

    print(f"CUDA version (torch built against): {torch.version.cuda}")
    print(f"Number of GPUs visible: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({vram_gb:.1f} GB VRAM)")

    # quick sanity op on GPU
    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("\nGPU matmul sanity check: OK")
    except Exception as e:
        print(f"\nGPU matmul sanity check FAILED: {e}")


def print_install_hint():
    print(
        "\nTo install the correct torch build for YOUR local CUDA driver:\n"
        "  1. Run `nvidia-smi` and note the 'CUDA Version' shown top-right\n"
        "     (this is the MAX CUDA version your driver supports, not what\n"
        "     you must match exactly — pick a torch CUDA build <= that number).\n"
        "  2. Go to https://pytorch.org/get-started/locally/ and select your OS,\n"
        "     pip, and the closest CUDA version <= what nvidia-smi reported.\n"
        "  3. Example (CUDA 12.1):\n"
        "     pip install torch torchvision torchaudio --index-url "
        "https://download.pytorch.org/whl/cu121\n"
        "  4. Re-run this script to confirm torch.cuda.is_available() is True.\n"
    )


def check_python_version():
    print("=" * 60)
    print("3. Python version")
    print("=" * 60)
    print(sys.version)
    print(
        "-> Confirm this matches what the rest of the team is using "
        "(agree on ONE version, e.g. 3.10 or 3.11)."
    )


if __name__ == "__main__":
    has_gpu = check_nvidia_driver()
    check_torch()
    check_python_version()
    print("=" * 60)
    if has_gpu:
        print("Done. Fix any warnings above before writing training code.")
    else:
        print("Done. No GPU detected — resolve this before proceeding to Phase 3.")
    print("=" * 60)
