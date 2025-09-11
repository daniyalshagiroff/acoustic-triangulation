import math
from typing import Dict, List, Tuple
from .contracts import Mic, ArrayConfig, Observation, Solution

def _bearing_from_pair(m1: Mic, m2: Mic, dt: float, c: float) -> Tuple[float, float]:
    """
    Возвращает (азимут_градусы, вес).
    Модель для пары микрофонов: d*sin(theta_rel) = c*dt,
    где d — расстояние между микрофонами, theta_rel — угол к базовой линии пары.
    Итоговый азимут = угол_базовой_линии + theta_rel (с корректной ориентацией).
    """
    x1,y1 = m1.xy; x2,y2 = m2.xy
    dx, dy = (x2-x1), (y2-y1)
    d = math.hypot(dx, dy)
    if d <= 1e-6:
        return (float("nan"), 0.0)

    # отношение пути
    arg = (c*dt)/max(d,1e-9)
    if abs(arg) > 1.0:
        # физически невозможное dt для данной пары -> обрежем и дадим низкий вес
        arg = max(-1.0, min(1.0, arg))
        weight = 0.2
    else:
        weight = 1.0

    theta_rel = math.degrees(math.asin(arg))  # [-90..+90] относительно baseline
    baseline_deg = math.degrees(math.atan2(dy, dx))  # 0° = +X, против часовой
    az = baseline_deg + theta_rel

    # нормализация 0..360
    az = (az % 360.0 + 360.0) % 360.0
    return az, weight

def estimate_azimuth(mics: Dict[str, Mic], cfg: ArrayConfig, obs: Observation) -> Solution:
    """
    Если указан ref_mic_id, используем пары (ref, other) по всем TDOA.
    Если нет — берём первую пару из obs как временную опорную.
    Возвращаем усреднённый по парам азимут (взвешенно).
    """
    if not obs.tdoa_s:
        return Solution(azimuth_deg=float("nan"), quality=0.0, used_mics=[])

    ref_id = cfg.ref_mic_id or next(iter(obs.tdoa_s.keys()))
    if ref_id not in mics:
        # если ref указан, но отсутствует — провал
        return Solution(azimuth_deg=float("nan"), quality=0.0, used_mics=[])

    az_list: List[Tuple[float,float]] = []
    used: List[str] = []
    for other_id, dt in obs.tdoa_s.items():
        if other_id == ref_id or other_id not in mics:
            continue
        az, w = _bearing_from_pair(mics[ref_id], mics[other_id], dt, cfg.speed_of_sound)
        if not math.isnan(az) and w>0:
            az_list.append((az, w))
            used.extend([ref_id, other_id])

    if not az_list:
        return Solution(azimuth_deg=float("nan"), quality=0.0, used_mics=[])

    # усреднение на окружности (через векторы)
    vx = sum(w*math.cos(math.radians(a)) for a,w in az_list)
    vy = sum(w*math.sin(math.radians(a)) for a,w in az_list)
    az_mean = math.degrees(math.atan2(vy, vx)) % 360.0

    # простая «quality»: длина результирующего вектора / сумма весов (0..1)
    w_sum = sum(w for _,w in az_list)
    r = math.hypot(vx, vy)
    quality = 0.0 if w_sum<=1e-9 else max(0.0, min(1.0, r / w_sum))

    return Solution(azimuth_deg=az_mean, quality=quality, used_mics=list(dict.fromkeys(used)))


