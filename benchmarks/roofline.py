#!/usr/bin/env python3
"""
roofline.py – Roofline analysis for ForgeRT kernels.

Plots achieved FLOPs vs arithmetic intensity, overlaid on the GPU's
compute and memory-bandwidth rooflines.

Usage:
    python benchmarks/roofline.py --gpu A100 --output roofline.png

If --gpu is not provided, the script reads `nvidia-smi` / `nvml` to get
peak bandwidth and compute.

Kernel data comes from:
  - bench_gemm output (copy results here manually or pipe with --gemm_csv)
  - bench_attention output (--attn_csv)
"""
import argparse
import sys
import math

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ---------------------------------------------------------------------------
# GPU roofline parameters (GFlops/s, GB/s)
# Add your GPU here.
# ---------------------------------------------------------------------------
GPU_SPECS = {
    "A100": {"peak_fp16_tflops": 312.0,   "peak_bw_gbs": 2000.0},
    "A10G": {"peak_fp16_tflops":  31.2,   "peak_bw_gbs":  600.0},
    "3090": {"peak_fp16_tflops":  35.6,   "peak_bw_gbs":  936.0},
    "4090": {"peak_fp16_tflops":  82.6,   "peak_bw_gbs": 1008.0},
    "H100": {"peak_fp16_tflops": 989.0,   "peak_bw_gbs": 3350.0},
}

# ---------------------------------------------------------------------------
# Kernel data: (name, arithmetic_intensity flops/byte, achieved_tflops)
# Fill in from bench_gemm and bench_attention runs.
# ---------------------------------------------------------------------------
KERNELS = [
    # (label, AI = FLOPs/byte, achieved_tflops)
    # --- decode GEMMs (M=1, low AI → memory bound) ---
    ("GEMV q_proj (decode)",     1.0,  0.01),   # PLACEHOLDER — fill from bench_gemm
    ("GEMV gate_proj (decode)",  1.0,  0.01),
    # --- prefill GEMMs (large M, high AI → compute bound) ---
    ("GEMM q_proj (prefill T=128)",   128.0, 10.0),
    ("GEMM q_proj (prefill T=512)",   512.0, 30.0),
    # --- decode attention (bandwidth bound) ---
    ("Attn v4 kv=512",   0.5, 0.001),
    ("Attn v4 kv=2048",  0.5, 0.001),
    ("Attn v4 kv=4096",  0.5, 0.001),
]


def ridge_point(peak_tflops, peak_bw_gbs) -> float:
    """AI at which compute and bandwidth ceilings meet."""
    return (peak_tflops * 1e12) / (peak_bw_gbs * 1e9)  # flops/byte


def draw_roofline(gpu_name: str, kernels, output_path: str):
    if not HAS_PLOT:
        print("matplotlib not installed; printing text roofline instead.")
        text_roofline(gpu_name, kernels)
        return

    spec = GPU_SPECS.get(gpu_name)
    if not spec:
        print(f"Unknown GPU '{gpu_name}'. Known: {list(GPU_SPECS.keys())}")
        sys.exit(1)

    peak_tflops = spec["peak_fp16_tflops"]
    peak_bw     = spec["peak_bw_gbs"]
    ridge_ai    = ridge_point(peak_tflops, peak_bw * 1e9 / 1e12)

    # AI range
    ai = np.logspace(-2, 4, 500)

    # Roofline: min(peak_bw * AI, peak_compute)
    roof_tflops = np.minimum(peak_bw * ai / 1e3, peak_tflops)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ai, roof_tflops, 'k-', linewidth=2, label='Roofline')
    ax.axvline(ridge_ai, color='gray', linestyle='--', alpha=0.5, label=f'Ridge ({ridge_ai:.0f} F/B)')

    # Annotate bandwidth slope
    ax.text(0.03, peak_bw * 0.03 / 1e3 * 1.5, f'{peak_bw} GB/s',
            fontsize=9, color='black', rotation=45)
    ax.axhline(peak_tflops, color='gray', linestyle=':', alpha=0.5)
    ax.text(200, peak_tflops * 1.05, f'{peak_tflops} TFLOPS', fontsize=9)

    # Plot kernels
    colors = plt.cm.tab10.colors
    for i, (name, ai_k, tflops_k) in enumerate(kernels):
        ax.scatter([ai_k], [tflops_k], s=80, color=colors[i % 10], zorder=5)
        ax.annotate(name, (ai_k, tflops_k),
                    textcoords="offset points", xytext=(8, 0), fontsize=7)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title(f"Roofline – {gpu_name} (FP16 Tensor Core)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved roofline plot to {output_path}")


def text_roofline(gpu_name: str, kernels):
    spec = GPU_SPECS.get(gpu_name, {"peak_fp16_tflops": "?", "peak_bw_gbs": "?"})
    print(f"\nRoofline – {gpu_name}")
    print(f"  Peak FP16:    {spec['peak_fp16_tflops']} TFLOPS")
    print(f"  Peak BW:      {spec['peak_bw_gbs']} GB/s")
    print(f"\n  {'Kernel':<35} {'AI (F/B)':>10} {'Achieved (TFLOPS)':>18} {'Bottleneck':>12}")
    print("  " + "-" * 80)
    if isinstance(spec['peak_fp16_tflops'], float):
        ridge = spec['peak_fp16_tflops'] * 1e12 / (spec['peak_bw_gbs'] * 1e9)
    else:
        ridge = float('inf')
    for name, ai, tflops in kernels:
        bn = "compute" if ai > ridge else "memory"
        print(f"  {name:<35} {ai:>10.1f} {tflops:>18.3f} {bn:>12}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",    default="A100")
    parser.add_argument("--output", default="roofline.png")
    args = parser.parse_args()

    draw_roofline(args.gpu, KERNELS, args.output)


if __name__ == "__main__":
    main()
