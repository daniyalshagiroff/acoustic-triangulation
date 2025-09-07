# Usage:
#   python tools/tdoa.py micA.wav micB.wav micC.wav \
#       --fs 16000 --frame_ms 32 --hop_ms 32 --max_tdoa_ms 2 \
#       --offsets_ms 0 120 -80 --jitter_ms 5 --print 10
#
# Требует: pip install soundfile numpy

import argparse, random, sys
import numpy as np
import soundfile as sf

def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)

def tiny_lag_sec(a: np.ndarray, b: np.ndarray, fs: int, max_ms: float) -> float:
    """Узкая кросс-корреляция: лаг b относительно a (сек). >0 => b позже a."""
    max_lag = int(round(max_ms * 1e-3 * fs))
    # предобработка
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
    ap = argparse.ArgumentParser(description="Compute TDOA (B-A, C-A) after timestamp alignment")
    ap.add_argument("wavs", nargs=3, help="micA.wav micB.wav micC.wav")
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--frame_ms", type=float, default=32)
    ap.add_argument("--hop_ms", type=float, default=32)
    ap.add_argument("--max_tdoa_ms", type=float, default=2.0, help="поиск лага ±ms")
    ap.add_argument("--offsets_ms", nargs=3, type=float, default=[0.0, 120.0, -80.0],
                    help="start-offsets для A B C в метках времени")
    ap.add_argument("--jitter_ms", type=float, default=5.0, help="±джиттер меток, мс")
    ap.add_argument("--print", dest="n_print", type=int, default=10, help="сколько окон вывести")
    args = ap.parse_args()

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

    # 4) считаем TDOA по каждому окну
    tdoa_BA, tdoa_CA = [], []
    for (A, B, C) in aligned:
        tdoa_BA.append(tiny_lag_sec(A, B, fs, args.max_tdoa_ms))  # B-A
        tdoa_CA.append(tiny_lag_sec(A, C, fs, args.max_tdoa_ms))  # C-A

    def stats(arr):
        a = np.array(arr, dtype=np.float64)
        return float(np.median(a)), float(np.mean(a)), float(np.std(a))

    mb, mab, sdb = stats(tdoa_BA)
    mc, mac, sdc = stats(tdoa_CA)

    print(f"[OK] Common aligned windows: {len(aligned)}  (frame={args.frame_ms} ms, hop={args.hop_ms} ms)")
    print(f"TDOA B-A (sec): median={mb:+.6f}  mean={mab:+.6f}  std={sdb:.6f}")
    print(f"TDOA C-A (sec): median={mc:+.6f}  mean={mac:+.6f}  std={sdc:.6f}\n")

    n = min(args.n_print, len(aligned))
    for i in range(n):
        print(f"win {i+1:02d}:  BA={tdoa_BA[i]:+.6f} s   CA={tdoa_CA[i]:+.6f} s")

if __name__ == "__main__":
    main()