# tools/sync_three_wav.py
# Usage:
#   python tools/sync_three_wav.py --ref 0 --out_dir aligned micA.wav micB.wav micC.wav
# Options:
#   --fs 16000           # целевой sample rate (ресемплинг при необходимости)
#   --max_tau 0.002      # макс. |задержка| на окне (сек) ~ d_max/c + запас
#   --interp 16          # интерполяция GCC-PHAT
#   --est_start 1.2      # старт окна (сек), где есть событие (выстрел/хлопок)
#   --est_dur 0.8        # длительность окна (сек)
#   --keep_len ref|max   # длина выходов: как у рефа (по умолч) или максимальная
#
# Requires: pip install pyroomacoustics soundfile scipy numpy

import argparse, os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import pyroomacoustics as pra  

def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)

def resample_if_needed(x: np.ndarray, fs: int, fs_target: int) -> tuple[np.ndarray, int]:
    if fs == fs_target:
        return x.astype(np.float32), fs
    from math import gcd
    g = gcd(fs_target, fs)
    up, down = fs_target // g, fs // g
    y = resample_poly(x, up, down).astype(np.float32)
    return y, fs_target

def fractional_shift_full(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """Сдвиг на дробное число отсчётов (может быть <0). Длина сохраняется (паддинг нулями)."""
    n = len(x)
    k = int(np.floor(shift_samples))
    frac = shift_samples - k

    # целая часть
    if k > 0:
        y = np.pad(x, (k, 0), mode="constant")[:n]
    elif k < 0:
        y = np.pad(x, (0, -k), mode="constant")[-k:n - k]
    else:
        y = x.copy()

    # дробная часть
    if abs(frac) < 1e-9:
        return y
    idx = np.arange(n, dtype=np.float64) - frac
    i0 = np.floor(idx).astype(int); i1 = i0 + 1
    w1 = idx - i0; w0 = 1.0 - w1
    valid0 = (i0 >= 0) & (i0 < n)
    valid1 = (i1 >= 0) & (i1 < n)
    z = np.zeros(n, dtype=np.float64)
    if np.any(valid0): z[valid0] += w0[valid0] * y[i0[valid0]]
    if np.any(valid1): z[valid1] += w1[valid1] * y[i1[valid1]]
    return z.astype(np.float32)

def gcc_phat(sig: np.ndarray, ref: np.ndarray, fs: int, max_tau: float, interp: int = 1) -> tuple[float, float]:
    """GCC-PHAT delay estimation between two signals."""
    # Ensure signals are the same length
    min_len = min(len(sig), len(ref))
    sig = sig[:min_len]
    ref = ref[:min_len]
    
    # Compute FFT
    N = len(sig)
    sig_fft = np.fft.fft(sig, n=N)
    ref_fft = np.fft.fft(ref, n=N)
    
    # Compute cross-power spectrum
    cross_power = sig_fft * np.conj(ref_fft)
    
    # PHAT weighting
    cross_power_phat = cross_power / (np.abs(cross_power) + 1e-10)
    
    # Compute inverse FFT
    correlation = np.fft.ifft(cross_power_phat)
    
    # Find peak
    max_tau_samples = int(max_tau * fs)
    search_range = min(max_tau_samples, N // 2)
    
    # Search in both positive and negative delays
    correlation_shifted = np.fft.fftshift(correlation)
    center = len(correlation_shifted) // 2
    start_idx = max(0, center - search_range)
    end_idx = min(len(correlation_shifted), center + search_range + 1)
    
    search_corr = correlation_shifted[start_idx:end_idx]
    peak_idx = np.argmax(np.abs(search_corr))
    
    # Convert to actual delay
    actual_peak_idx = start_idx + peak_idx - center
    delay_samples = actual_peak_idx
    
    # Convert to seconds
    delay_sec = delay_samples / fs
    
    # Get correlation value
    corr_value = float(np.abs(search_corr[peak_idx]))
    
    return delay_sec, corr_value

def estimate_delay(sig: np.ndarray, ref: np.ndarray, fs: int, max_tau: float, interp: int) -> float:
    """Возвращает задержку sig относительно ref (сек). >0 => sig ПОЗЖЕ ref."""
    tau, _ = gcc_phat(sig.astype(np.float32), ref.astype(np.float32),
                      fs=fs, max_tau=max_tau, interp=interp)
    return float(tau)

def align_to_reference(signals: list[np.ndarray], fs: int, ref_idx: int,
                       max_tau: float, interp: int,
                       est_start_s: float | None, est_dur_s: float | None,
                       keep_len: str = "ref") -> tuple[list[np.ndarray], list[float]]:
    """Выровнять три дорожки к ref. Окно est_* только для оценки задержки; сами треки не режутся."""
    ref = signals[ref_idx]
    N_ref = len(ref)
    if est_dur_s is None: est_dur_s = 1.0
    if est_start_s is None: est_start_s = max(0.0, (N_ref / fs) / 2 - est_dur_s / 2)

    est_start = int(est_start_s * fs)
    est_len   = int(est_dur_s * fs)
    est_end   = min(N_ref, est_start + est_len)
    ref_est   = ref[est_start:est_end]

    delays = [0.0] * len(signals)
    aligned = [None] * len(signals)
    aligned[ref_idx] = ref.copy()

    # 1) оценка задержек на окне
    for i, s in enumerate(signals):
        if i == ref_idx: continue
        s_est = s[est_start:est_end]
        if len(s_est) >= 32 and len(ref_est) >= 32:
            delays[i] = estimate_delay(s_est, ref_est, fs, max_tau, interp)
        else:
            delays[i] = 0.0  # окно слишком короткое — пропускаем

    # 2) применяем сдвиг ко всей дорожке (длина сохраняется)
    for i, s in enumerate(signals):
        if i == ref_idx: continue
        shift_samples = -delays[i] * fs  # инвертируем знак, чтобы совместить с ref
        aligned[i] = fractional_shift_full(s, shift_samples)

    # 3) выравниваем длину выходов
    target_len = len(signals[ref_idx]) if keep_len == "ref" else max(len(a) for a in aligned)
    out = []
    for a in aligned:
        if len(a) < target_len:
            a = np.pad(a, (0, target_len - len(a)), mode="constant")
        else:
            a = a[:target_len]
        out.append(a)
    return out, delays

def main():
    ap = argparse.ArgumentParser(description="Sync 3 WAVs via GCC-PHAT to a reference")
    ap.add_argument("wavs", nargs=3, help="Paths to 3 wav files (micA micB micC)")
    ap.add_argument("--ref", type=int, default=0, help="Reference index: 0/1/2 (default 0)")
    ap.add_argument("--fs", type=int, default=16000, help="Target sample rate (resample if needed)")
    ap.add_argument("--max_tau", type=float, default=0.002, help="Max |delay| (sec), e.g. d_max/c + margin")
    ap.add_argument("--interp", type=int, default=16, help="GCC-PHAT interpolation factor")
    ap.add_argument("--est_start", type=float, default=None, help="Window start (sec) for delay estimation")
    ap.add_argument("--est_dur", type=float, default=None, help="Window duration (sec) for delay estimation")
    ap.add_argument("--keep_len", choices=["ref","max"], default="ref",
                    help='Output length: "ref" (default) or "max"')
    ap.add_argument("--out_dir", type=str, default="aligned", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # загрузка и приведение к общему fs
    signals, names = [], []
    for p in args.wavs:
        x, fs = sf.read(p, always_2d=False)
        x = to_mono(x).astype(np.float32)
        x, _ = resample_if_needed(x, fs, args.fs)
        signals.append(x); names.append(os.path.splitext(os.path.basename(p))[0])

    aligned, delays = align_to_reference(
        signals, fs=args.fs, ref_idx=args.ref,
        max_tau=args.max_tau, interp=args.interp,
        est_start_s=args.est_start, est_dur_s=args.est_dur,
        keep_len=args.keep_len
    )

    # сохранить
    for i, y in enumerate(aligned):
        outp = os.path.join(args.out_dir, f"{names[i]}_aligned_ref{args.ref}.wav")
        sf.write(outp, y, args.fs)
        print(f"[OK] saved: {outp}")

    # печать задержек
    print("\nDetected delays relative to reference (seconds):")
    for i, d in enumerate(delays):
        tag = "(ref)" if i == args.ref else ""
        print(f"  track {i}: {d:+.6f} s {tag}")

if __name__ == "__main__":
    main()