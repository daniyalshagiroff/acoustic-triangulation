# tools/gen_from_source.py
# Usage:
#   python gen_from_source.py source.wav
# Creates: micA.wav (без сдвига), micB.wav (+0.00040 s позже A), micC.wav (-0.00020 s раньше A)

import sys
import numpy as np
import soundfile as sf

def fractional_shift_full(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """
    Сдвиг на дробное число отсчётов с сохранением полной длины (паддинг нулями).
    shift_samples > 0  => сигнал позже (сдвигаем вправо, в начало паддинг)
    shift_samples < 0  => сигнал раньше (сдвигаем влево, паддинг в конец)
    """
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

    if abs(frac) < 1e-9:
        return y

    # дробная часть (линейная интерполяция), тоже без изменения длины
    idx = np.arange(n, dtype=np.float64) - frac
    i0 = np.floor(idx).astype(int)
    i1 = i0 + 1
    w1 = idx - i0
    w0 = 1.0 - w1

    valid0 = (i0 >= 0) & (i0 < n)
    valid1 = (i1 >= 0) & (i1 < n)

    z = np.zeros(n, dtype=np.float64)
    if np.any(valid0): z[valid0] += w0[valid0] * y[i0[valid0]]
    if np.any(valid1): z[valid1] += w1[valid1] * y[i1[valid1]]
    return z.astype(np.float32)

def to_mono(x):
    return x if x.ndim == 1 else np.mean(x, axis=1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python gen_from_source.py source.wav")
        sys.exit(1)

    src_path = sys.argv[1]
    x, fs = sf.read(src_path, always_2d=False)
    x = to_mono(x).astype(np.float32)
    n = len(x)

    # Задаём задержки (секунды): B позже A, C раньше A
    delay_B_s = +0.00040
    delay_C_s = -0.00020

    A = x
    B = fractional_shift_full(x, delay_B_s * fs)
    C = fractional_shift_full(x, delay_C_s * fs)

    # сохраняем — длина у всех равна исходной (например, 4 секунды)
    sf.write("micA.wav", A, fs)
    sf.write("micB.wav", B, fs)
    sf.write("micC.wav", C, fs)
    print(f"Saved micA/B/C.wav at fs={fs}, length={n/fs:.3f}s")

if __name__ == "__main__":
    main()