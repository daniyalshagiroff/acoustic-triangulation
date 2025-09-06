from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class Mic:
    id: str
    xy: Tuple[float, float]  # метры, (x,y) в единой плоскости

@dataclass
class ArrayConfig:
    speed_of_sound: float = 343.0  # м/с (20°C)
    ref_mic_id: Optional[str] = None  # опорный микрофон (для TDOA)

@dataclass
class Observation:
    # TDOA относительно ref_mic_id: { "mic_id": delta_t_seconds }
    # напр.: {"micB": 0.0008, "micC": -0.0002}
    tdoa_s: Dict[str, float]

@dataclass
class Solution:
    azimuth_deg: float            # оценка направления 0..360 (0 = +X ось)
    quality: float                # 0..1 условная метрика согласованности
    used_mics: List[str]          # какие пары/мики учтены