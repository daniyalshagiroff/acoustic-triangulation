# Usage:
#   python tools/align_wavs.py micA.wav micB.wav micC.wav
#
# Требует: pip install soundfile numpy

import sys, random
import numpy as np
import soundfile as sf

# -------------------- ПАРАМЕТРЫ (меняй при желании) --------------------
FS_TARGET = 16000        # целевой sample rate (для простоты требуем одинаковый)
FRAME_MS  = 32           # длина кадра (например, 32 мс)
HOP_MS    = 32           # шаг между кадрами (можно = FRAME_MS)
# «Большие» стартовые оффсеты ТОЛЬКО в метках времени (секунды):
START_OFFSETS_MS = [0.0, 120.0, -80.0]   # A:0ms, B:+120ms, C:-80ms
JITTER_MS_RANGE  = 5.0   # случайный «сетевой» джиттер в таймстампах ±5мс
# Для оценки микро-лагов внутри кадра (узкое окно вокруг 0):
MAX_TDOA_MS      = 2.0   # искать лаги ±2мс (должно хватить для реальных расстояний)
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
    if len(sys.argv) != 4:
        print("Usage: python tools/ts_align_three_wav.py micA.wav micB.wav micC.wav")
        sys.exit(1)

    # 1) загрузка трёх файлов (моно), проверка sample rate
    paths = sys.argv[1:]
    signals = []
    rates = []
    for p in paths:
        x, fs = sf.read(p, always_2d=False)
        x = to_mono(np.asarray(x, dtype=np.float32))
        signals.append(x); rates.append(fs)

    if len(set(rates)) != 1 or rates[0] != FS_TARGET:
        print(f"[!] For simplicity this demo expects all WAVs to have the same sample rate = {FS_TARGET} Hz.")
        print(f"    Your rates: {rates}. Resample your files first (или поменяй FS_TARGET и убери эту проверку).")
        sys.exit(1)

    fs = rates[0]
    frame = int(round(FRAME_MS * 1e-3 * fs))
    hop   = int(round(HOP_MS   * 1e-3 * fs))

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
            jitter = (random.random()*2 - 1) * (JITTER_MS_RANGE * 1e-3)   # ±джиттер в сек
            ts_utc = t0 + (start_offset_ms * 1e-3) + k * (hop / fs) + jitter
            stream.append((ts_utc, chunk))
        return stream

    streams = [make_stream(signals[i], START_OFFSETS_MS[i]) for i in range(3)]

    # 3) Джиттер-буфер: собираем кадры по общим временным слотам (квант = hop/fs)
    slot = hop / fs  # длительность слота в сек
    buckets = {}     # slot_key -> { mic_index: chunk }
    for mic_idx, stream in enumerate(streams):
        for ts_utc, chunk in stream:
            key = round(ts_utc / slot)  # привязываем каждый кадр к ближайшему «слоту»
            if key not in buckets:
                buckets[key] = {}
            buckets[key][mic_idx] = chunk

    # 4) Формируем выровненные по времени окна: берем только те слоты, где есть все три микрофона
    aligned_windows = []  # список кортежей (A_chunk, B_chunk, C_chunk)
    for key in sorted(buckets.keys()):
        row = buckets[key]
        if all(i in row for i in (0,1,2)):   # у всех 3 присутствуют
            aligned_windows.append((row[0], row[1], row[2]))

    if not aligned_windows:
        print("[!] Не получилось собрать общий слот для всех 3 микрофонов. Увеличь длину файлов или уменьши JITTER/увеличь FRAME/HOP.")
        sys.exit(1)

    print(f"[OK] Собрано общих временных окон: {len(aligned_windows)}")
    print("Мы НИ РАЗУ не сдвигали/не редактировали аудио — только сгруппировали кадры по одинаковым меткам времени.")
    print("Теперь проверим микро-TDOA внутри первых окон (они должны сохраниться):\n")

    # 5) Для первых N окон посчитаем микро-лаг B-A и C-A (в секундах)
    max_tdoa_s = MAX_TDOA_MS * 1e-3
    for i, (A, B, C) in enumerate(aligned_windows[:N_PRINT], start=1):
        tau_BA = tiny_lag_sec(A, B, fs, MAX_TDOA_MS)  # >0 => B позже A
        tau_CA = tiny_lag_sec(A, C, fs, MAX_TDOA_MS)  # >0 => C позже A
        print(f"Окно {i:02d}:  TDOA(B-A) = {tau_BA:+.6f} s   TDOA(C-A) = {tau_CA:+.6f} s")

    print("\nВывод:")
    print("- Мы «синхронизировали по времени» только метками (слотами),")
    print("- но не выравнивали аудио по пикам, поэтому микро-TDOA (сотни микросекунд) остались —")
    print("  именно они и нужны тебе для последующей триангуляции.")
    print("\nПодсказка: чтобы смоделировать другие условия, поменяй START_OFFSETS_MS, JITTER_MS_RANGE, FRAME_MS.")
    print("В реальном стриме роль «слотов» выполняет твой джиттер-буфер, который собирает кадры по общей оси времени.")
    
if __name__ == "__main__":
    main()