"""
Usage:
  python tools/gen_from_source.py path/to/source.wav --out_dir tools

Creates three WAVs at 16 kHz mono with sub-sample delays:
  micA.wav  (baseline)
  micB.wav  (+0.00040 s later than A)
  micC.wav  (-0.00020 s earlier than A)
"""

import argparse
import os
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_FS = 16000
DELAY_B_S = +0.00040
DELAY_C_S = -0.00020


def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)


def resample_if_needed(x: np.ndarray, fs: int, fs_target: int) -> tuple[np.ndarray, int]:
    if fs == fs_target:
        return x.astype(np.float32), fs
    g = gcd(fs_target, fs)
    up, down = fs_target // g, fs // g
    y = resample_poly(x, up, down).astype(np.float32)
    return y, fs_target


def fractional_shift_full(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """Shift signal by fractional samples; keep original length via zero padding."""
    n = len(x)
    k = int(np.floor(shift_samples))
    frac = shift_samples - k

    # integer part
    if k > 0:
        y = np.pad(x, (k, 0), mode="constant")[:n]
    elif k < 0:
        y = np.pad(x, (0, -k), mode="constant")[-k:n - k]
    else:
        y = x.copy()

    # fractional part (linear interpolation)
    if abs(frac) < 1e-12:
        return y.astype(np.float32)
    idx = np.arange(n, dtype=np.float64) - frac
    i0 = np.floor(idx).astype(int)
    i1 = i0 + 1
    w1 = idx - i0
    w0 = 1.0 - w1
    valid0 = (i0 >= 0) & (i0 < n)
    valid1 = (i1 >= 0) & (i1 < n)
    z = np.zeros(n, dtype=np.float64)
    if np.any(valid0):
        z[valid0] += w0[valid0] * y[i0[valid0]]
    if np.any(valid1):
        z[valid1] += w1[valid1] * y[i1[valid1]]
    return z.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 3 test mic WAVs with micro-delays at 16 kHz mono")
    ap.add_argument("source", help="Path to source WAV")
    ap.add_argument("--out_dir", default="tools", help="Output directory (default: tools)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    x, fs = sf.read(args.source, always_2d=False)
    x = to_mono(np.asarray(x, dtype=np.float32))
    x, fs = resample_if_needed(x, fs, TARGET_FS)

    A = x
    B = fractional_shift_full(x, DELAY_B_S * fs)
    C = fractional_shift_full(x, DELAY_C_S * fs)

    sf.write(os.path.join(args.out_dir, "micA.wav"), A, fs)
    sf.write(os.path.join(args.out_dir, "micB.wav"), B, fs)
    sf.write(os.path.join(args.out_dir, "micC.wav"), C, fs)
    print(f"[OK] Saved A/B/C at {fs} Hz, length={len(A)/fs:.3f}s with delays: B={DELAY_B_S:+.6f}s C={DELAY_C_S:+.6f}s")


if __name__ == "__main__":
    main()


