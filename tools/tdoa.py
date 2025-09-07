# Usage (из WAV напрямую):
#   python tools/tdoa.py micA.wav micB.wav micC.wav \
#       --fs 16000 --frame_ms 32 --hop_ms 32 --max_tdoa_ms 2 \
#       --offsets_ms 0 120 -80 --jitter_ms 5 --print 10
#
# Usage (из NPZ, созданного align_wavs.py):
#   python tools/tdoa.py --from_npz tools/aligned_windows.npz --max_tdoa_ms 2 --print 10
#
# Требует: pip install soundfile numpy

import argparse, random, sys, os
import numpy as np
import soundfile as sf
try:
    import pyroomacoustics as pra
except Exception:
    pra = None

def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)

def tiny_lag_sec(a: np.ndarray, b: np.ndarray, fs: int, max_ms: float) -> float:
    """Оценка лага: GCC-PHAT если доступен, иначе узкая корреляция. >0 => b позже a."""
    # Хэннинг окно для снижения утечек
    if len(a) == len(b):
        win = np.hanning(len(a)).astype(np.float32)
        a = (a * win).astype(np.float32)
        b = (b * win).astype(np.float32)

    if pra is not None and hasattr(pra, 'gcc_phat'):
        # GCC-PHAT: чтобы tau > 0 означало "b позже a", вызываем как (b, a)
        tau, _ = pra.gcc_phat(b.astype(np.float32), a.astype(np.float32), fs=fs,
                              max_tau=max_ms*1e-3, interp=32)
        return float(tau)
    # fallback: узкая корреляция
    max_lag = int(round(max_ms * 1e-3 * fs))
    a = a - np.mean(a); b = b - np.mean(b)
    ma = np.max(np.abs(a)) or 1.0
    mb = np.max(np.abs(b)) or 1.0
    a = a / ma; b = b / mb
    best_val, best_lag = -1e9, 0
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            aa = a[:len(a)-L]; bb = b[L:]
        else:
            aa = a[-L:];       bb = b[:len(b)+L]
        if len(aa) == 0:
            continue
        val = float(np.sum(aa * bb))
        if val > best_val:
            best_val, best_lag = val, L
    return best_lag / float(fs)

def main():
    ap = argparse.ArgumentParser(description="Compute TDOA (B-A, C-A) from WAVs or NPZ aligned windows")
    ap.add_argument("wavs", nargs="*", help="micA.wav micB.wav micC.wav")
    ap.add_argument("--from_npz", help="Path to NPZ with aligned windows (from align_wavs.py)")
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--frame_ms", type=float, default=32)
    ap.add_argument("--hop_ms", type=float, default=32)
    ap.add_argument("--max_tdoa_ms", type=float, default=None, help="поиск лага ±ms; если не задано, вычислим из d_max (и ограничим 1.0)")
    ap.add_argument("--d_max_m", type=float, default=0.3, help="макс. межмикрофонное расстояние (м)")
    ap.add_argument("--offsets_ms", nargs=3, type=float, default=[0.0, 120.0, -80.0],
                    help="start-offsets для A B C в метках времени (только для WAV режима)")
    ap.add_argument("--jitter_ms", type=float, default=5.0, help="±джиттер меток, мс (только для WAV режима)")
    ap.add_argument("--print", dest="n_print", type=int, default=10, help="сколько окон вывести")
    args = ap.parse_args()

    if args.from_npz:
        data = np.load(args.from_npz)
        A = data["A"]; B = data["B"]; C = data["C"]
        fs = int(data["fs"]) if "fs" in data else args.fs
        aligned = list(zip(A, B, C))
    else:
        if len(args.wavs) != 3:
            print("Usage: either provide 3 WAVs or use --from_npz path.npz")
            sys.exit(1)
        # загрузка
        sigs, rates = [], []
        for p in args.wavs:
            x, fs = sf.read(p, always_2d=False)
            x = to_mono(np.asarray(x, dtype=np.float32))
            sigs.append(x); rates.append(fs)
        if len(set(rates)) != 1 or rates[0] != args.fs:
            print(f"[!] Expect same sample rate = {args.fs} Hz. Got: {rates}. Resample first."); sys.exit(1)

        fs = args.fs
        frame = int(round(args.frame_ms * 1e-3 * fs))
        hop   = int(round(args.hop_ms   * 1e-3 * fs))

    # 1) формируем «потоки» (метки времени + кусочки), аудио не меняем
    def make_stream(x: np.ndarray, start_offset_ms: float):
        t0 = 1_000_000.0
        n_frames = max(0, (len(x) - frame) // hop + 1)
        stream = []
        for k in range(n_frames):
            beg, end = k * hop, k * hop + frame
            if end > len(x): break
            chunk = x[beg:end]
            jitter = (random.random()*2 - 1) * (args.jitter_ms * 1e-3)
            ts = t0 + start_offset_ms*1e-3 + k*(hop/fs) + jitter
            stream.append((ts, chunk))
        return stream

    if not args.from_npz:
        streams = [make_stream(sigs[i], args.offsets_ms[i]) for i in range(3)]

        # 2) «джиттер-буфер»: группируем по временной сетке
        slot = hop / fs
        buckets = {}  # key -> {mic_idx: chunk}
        for mic_idx, stream in enumerate(streams):
            for ts, chunk in stream:
                key = round(ts / slot)
                buckets.setdefault(key, {})[mic_idx] = chunk

        # 3) берём только слоты, где есть все три микрофона
        aligned = []
        for key in sorted(buckets.keys()):
            row = buckets[key]
            if all(i in row for i in (0,1,2)):
                aligned.append((row[0], row[1], row[2]))

    if not aligned:
        print("[!] Нет общих окон. Увеличь длину файлов или уменьшай jitter/увеличь frame."); sys.exit(1)

    # 4) энерго-фильтр окон + считаем TDOA по каждому окну
    # адаптивное окно поиска
    if args.max_tdoa_ms is None:
        max_ms = min(1.0, 1000.0 * args.d_max_m / 343.0)
    else:
        max_ms = min(1.0, float(args.max_tdoa_ms))

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))

    # посчитаем RMS по всем окнам (A B C вместе) и установим порог на 60 перцентиле
    energies = []
    for (Awin, Bwin, Cwin) in aligned:
        energies.append(max(rms(Awin), rms(Bwin), rms(Cwin)))
    if not energies:
        print("[!] Нет окон для оценки")
        sys.exit(1)
    thr = float(np.percentile(np.array(energies, dtype=np.float64), 60.0))

    tdoa_BA, tdoa_CA = [], []
    max_tau_sec = max_ms * 1e-3
    for (Awin, Bwin, Cwin) in aligned:
        if max(rms(Awin), rms(Bwin), rms(Cwin)) < thr:
            continue
        tau_BA = tiny_lag_sec(Awin, Bwin, fs, max_ms)  # B-A
        tau_CA = tiny_lag_sec(Awin, Cwin, fs, max_ms)  # C-A
        # отбрасываем окна, где оценка "упёрлась" в край диапазона
        if abs(tau_BA) >= 0.98 * max_tau_sec or abs(tau_CA) >= 0.98 * max_tau_sec:
            continue
        tdoa_BA.append(tau_BA)
        tdoa_CA.append(tau_CA)

    def stats(arr):
        a = np.array(arr, dtype=np.float64)
        return float(np.median(a)), float(np.mean(a)), float(np.std(a))

    # 5) робастная агрегация: обрежем 10% хвостов (IQR-like) и посчитаем медиану/среднее/стд
    def trim(arr, p=10.0):
        if not arr:
            return np.array([], dtype=np.float64)
        a = np.sort(np.array(arr, dtype=np.float64))
        n = len(a)
        k = int(np.floor(n * p / 100.0))
        return a[k:n-k] if n - 2*k > 0 else a

    BA_t = trim(tdoa_BA, 10.0)
    CA_t = trim(tdoa_CA, 10.0)

    def stats(a):
        if a.size == 0:
            return (float('nan'), float('nan'), float('nan'))
        return (float(np.median(a)), float(np.mean(a)), float(np.std(a)))

    mb, mab, sdb = stats(BA_t)
    mc, mac, sdc = stats(CA_t)

    print(f"[OK] Common aligned windows: {len(aligned)}  (kept after energy filter: {len(BA_t) if np.isfinite(mb) else 0})")
    print(f"TDOA B-A (sec): median={mb:+.6f}  mean={mab:+.6f}  std={sdb:.6f}")
    print(f"TDOA C-A (sec): median={mc:+.6f}  mean={mac:+.6f}  std={sdc:.6f}\n")

    n = min(args.n_print, len(aligned))
    for i in range(n):
        print(f"win {i+1:02d}:  BA={tdoa_BA[i]:+.6f} s   CA={tdoa_CA[i]:+.6f} s")

if __name__ == "__main__":
    main()