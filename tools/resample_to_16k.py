import argparse
import os
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_FS = 16000


def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)


def resample_if_needed(x: np.ndarray, fs: int, fs_target: int) -> tuple[np.ndarray, int]:
    if fs == fs_target:
        return x.astype(np.float32), fs
    g = gcd(fs_target, fs)
    up, down = fs_target // g, fs // g
    y = resample_poly(x, up, down).astype(np.float32)
    return y, fs_target


def main() -> None:
    ap = argparse.ArgumentParser(description="Resample WAV files to 16 kHz")
    ap.add_argument("wavs", nargs="+", help="Paths to WAV files to resample")
    ap.add_argument("--out_dir", default="tools/16k", help="Output directory (default: tools/16k)")
    ap.add_argument("--mono", action="store_true", help="Convert to mono (average channels)")
    ap.add_argument("--keep_name", action="store_true", help="Keep original file name (no _16k suffix)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for p in args.wavs:
        x, fs = sf.read(p, always_2d=False)
        if args.mono:
            x = to_mono(np.asarray(x, dtype=np.float32))
        else:
            x = np.asarray(x, dtype=np.float32)

        # If stereo and not mono, resample each channel independently
        if x.ndim == 2:
            channels = []
            for ch in range(x.shape[1]):
                y, _ = resample_if_needed(x[:, ch], fs, TARGET_FS)
                channels.append(y)
            y = np.stack(channels, axis=1)
        else:
            y, _ = resample_if_needed(x, fs, TARGET_FS)

        base = os.path.splitext(os.path.basename(p))[0]
        out_name = f"{base}.wav" if args.keep_name else f"{base}_16k.wav"
        out_path = os.path.join(args.out_dir, out_name)
        sf.write(out_path, y, TARGET_FS)
        print(f"[OK] {p} -> {out_path} @ {TARGET_FS} Hz, shape={y.shape}")


if __name__ == "__main__":
    main()


