# Usage:
#   python tools/align_wavs.py micA.wav micB.wav micC.wav --out_npz tools/aligned_windows.npz
#
# Что делает:
# - Читает три WAV, режет на окна (frame, hop)
# - Строит «слоты» по меткам времени, убирая большие сдвиги и джиттер
# - НЕ меняет аудио, только группирует совпадающие по времени окна
# - На выходе сохраняет aligned windows в NPZ: A (N, frame), B (N, frame), C (N, frame), fs, frame, hop
#
# Требует: pip install soundfile numpy

import sys, random, argparse, os
import numpy as np
import soundfile as sf

# -------------------- ПАРАМЕТРЫ (меняй при желании) --------------------
FS_TARGET = 16000        # целевой sample rate (для простоты требуем одинаковый)
FRAME_MS  = 64           # длина кадра (например, 64 мс)
HOP_MS    = 32           # шаг между кадрами (перекрытие 50%)
# «Большие» стартовые оффсеты ТОЛЬКО в метках времени (секунды):
START_OFFSETS_MS = [0.0, 120.0, -80.0]   # A:0ms, B:+120ms, C:-80ms
JITTER_MS_RANGE  = 0   # случайный «сетевой» джиттер в таймстампах ±5мс
# Для оценки микро-лагов внутри кадра (узкое окно вокруг 0):
MAX_TDOA_MS      = 1.0   # искать лаги ±2мс (должно хватить для реальных расстояний)
N_PRINT           = 10    # сколько первых слотов распечатать
# -----------------------------------------------------------------------

def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)

def tiny_lag_sec(a: np.ndarray, b: np.ndarray, fs: int, max_ms: float) -> float:
    """
    Очень простая оценка лага между a и b по кросс-корреляции в узком диапазоне.
    Возвращает лаг (сек), >0 значит b ПОЗЖЕ a.
    """
    max_lag = int(round(max_ms * 1e-3 * fs))
    # нормализация, чтобы корреляция не «плавала»
    a = a - np.mean(a); b = b - np.mean(b)
    da = np.max(np.abs(a)) or 1.0
    db = np.max(np.abs(b)) or 1.0
    a = a / da; b = b / db

    # считаем корреляцию вручную в диапазоне [-max_lag, +max_lag]
    lags = range(-max_lag, max_lag + 1)
    best_lag = 0
    best_val = -1e9
    for L in lags:
        if L >= 0:
            aa = a[:len(a)-L]
            bb = b[L:]
        else:
            aa = a[-L:]
            bb = b[:len(b)+L]
        if len(aa) == 0: 
            continue
        val = float(np.sum(aa * bb))  # простая мера похожести
        if val > best_val:
            best_val = val
            best_lag = L
    return best_lag / float(fs)

def main():
    ap = argparse.ArgumentParser(description="Align three WAV streams into timestamp-aligned windows and dump to NPZ")
    ap.add_argument("wavs", nargs=3, help="micA.wav micB.wav micC.wav")
    ap.add_argument("--fs", type=int, default=FS_TARGET, help="Expected sample rate (all must match)")
    ap.add_argument("--frame_ms", type=float, default=FRAME_MS)
    ap.add_argument("--hop_ms", type=float, default=HOP_MS)
    ap.add_argument("--offsets_ms", nargs=3, type=float, default=START_OFFSETS_MS,
                    help="start-offsets для A B C в метках времени")
    ap.add_argument("--jitter_ms", type=float, default=JITTER_MS_RANGE, help="±джиттер меток, мс")
    ap.add_argument("--out_npz", required=True, help="Path to save aligned windows NPZ")
    ap.add_argument("--n_print", type=int, default=N_PRINT, help="Сколько окон вывести для проверки")
    args = ap.parse_args()

    # 1) загрузка трёх файлов (моно), проверка sample rate
    paths = args.wavs
    signals = []
    rates = []
    for p in paths:
        x, fs = sf.read(p, always_2d=False)
        x = to_mono(np.asarray(x, dtype=np.float32))
        signals.append(x); rates.append(fs)

    if len(set(rates)) != 1 or rates[0] != args.fs:
        print(f"[!] For simplicity this demo expects all WAVs to have the same sample rate = {args.fs} Hz.")
        print(f"    Your rates: {rates}. Resample your files first (или поменяй --fs/пересэмпли).")
        sys.exit(1)

    fs = rates[0]
    frame = int(round(args.frame_ms * 1e-3 * fs))
    hop   = int(round(args.hop_ms   * 1e-3 * fs))

    # 2) «Стрим» с таймстампами: НЕ трогаем аудио, только создаём метки времени с оффсетами и джиттером
    #    представим, что все три девайса шлют чанки одинаковой длины, но начали в разное время и с джиттером
    def make_stream(x: np.ndarray, start_offset_ms: float):
        stream = []
        # «глобальное» начало времени (UTC для демонстрации)
        t0 = 1_000_000.0  # произвольное число секунд
        n_frames = max(0, (len(x) - frame) // hop + 1)
        for k in range(n_frames):
            beg = k * hop
            end = beg + frame
            if end > len(x): break
            chunk = x[beg:end]
            jitter = (random.random()*2 - 1) * (args.jitter_ms * 1e-3)   # ±джиттер в сек
            ts_utc = t0 + (start_offset_ms * 1e-3) + k * (hop / fs) + jitter
            stream.append((ts_utc, chunk))
        return stream

    streams = [make_stream(signals[i], float(args.offsets_ms[i])) for i in range(3)]

    # 3) Робастное построение слотов: явная сетка k*slot и подбор ближайшего чанка с допуском slot/2
    slot = hop / fs
    # собираем все времена для оценки диапазона
    all_ts = []
    for stream in streams:
        if stream:
            all_ts.extend(ts for ts,_ in stream)
    if not all_ts:
        print("[!] Нет чанков во входных потоках")
        sys.exit(1)
    t_min = min(all_ts); t_max = max(all_ts)
    k_start = int(np.floor(t_min / slot))
    k_end   = int(np.ceil(t_max / slot))

    def nearest_within(stream, t_target, tol):
        best = None; best_dt = 1e9
        for ts, chunk in stream:
            dt = abs(ts - t_target)
            if dt <= tol and dt < best_dt:
                best = chunk; best_dt = dt
        return best

    aligned_windows = []
    tol = 0.5 * slot
    for k in range(k_start, k_end + 1):
        t_center = k * slot
        picks = []
        for mic_idx in range(3):
            chunk = nearest_within(streams[mic_idx], t_center, tol)
            picks.append(chunk)
        if all(ch is not None for ch in picks):
            aligned_windows.append(tuple(picks))

    if not aligned_windows:
        print("[!] Не получилось собрать общий слот для всех 3 микрофонов. Увеличь длину файлов или уменьши JITTER/увеличь FRAME/HOP.")
        sys.exit(1)

    print(f"[OK] Собрано общих временных окон: {len(aligned_windows)}")
    print("Мы НИ РАЗУ не сдвигали/не редактировали аудио — только сгруппировали кадры по одинаковым меткам времени.")

    # 5) Сохраняем в NPZ для последующего TDOA
    A = np.stack([w[0] for w in aligned_windows], axis=0)
    B = np.stack([w[1] for w in aligned_windows], axis=0)
    C = np.stack([w[2] for w in aligned_windows], axis=0)
    os.makedirs(os.path.dirname(args.out_npz) or '.', exist_ok=True)
    np.savez_compressed(args.out_npz, A=A, B=B, C=C, fs=fs, frame=frame, hop=hop)
    print(f"[OK] Dumped aligned windows to: {args.out_npz}  (A,B,C shape: {A.shape})\n")

    # 6) Для визуальной проверки (опционально)
    for i, (Aa, Bb, Cc) in enumerate(aligned_windows[:args.n_print], start=1):
        tau_BA = tiny_lag_sec(Aa, Bb, fs, MAX_TDOA_MS)  # >0 => B позже A
        tau_CA = tiny_lag_sec(Aa, Cc, fs, MAX_TDOA_MS)  # >0 => C позже A
        print(f"Окно {i:02d}:  TDOA(B-A) = {tau_BA:+.6f} s   TDOA(C-A) = {tau_CA:+.6f} s")

    print("\nВывод:")
    print("- Мы «синхронизировали по времени» только метками (слотами),")
    print("- но не выравнивали аудио по пикам, поэтому микро-TDOA (сотни микросекунд) остались —")
    print("  именно они и нужны тебе для последующей триангуляции.")
    print("\nПодсказка: чтобы смоделировать другие условия, поменяй START_OFFSETS_MS, JITTER_MS_RANGE, FRAME_MS.")
    print("В реальном стриме роль «слотов» выполняет твой джиттер-буфер, который собирает кадры по общей оси времени.")
    
if __name__ == "__main__":
    main()