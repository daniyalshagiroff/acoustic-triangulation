"""
Acoustic triangulation package for TDOA-based azimuth calculation.
"""

from triangulation.contracts import Mic, ArrayConfig, Observation
from triangulation.solver import estimate_azimuth

# 1) геометрия
mics_cfg = {
    "A": Mic(id="A", xy=(0.00, 0.00)),
    "B": Mic(id="B", xy=(0.30, 0.00)),
    "C": Mic(id="C", xy=(0.00, 0.30)),
}

# 2) настройки
cfg = ArrayConfig(speed_of_sound=343.0, ref_mic_id="A")  # 343 м/с около 20°C

# 3) TDOA относительно A (секунды)
obs = Observation(tdoa_s={
    "B": +0.00040,
    "C": -0.00020,
})

# 4) оценка азимута
sol = estimate_azimuth(mics_cfg, cfg, obs)
print({
    "azimuth_deg": round(sol.azimuth_deg, 2),
    "quality": round(sol.quality, 3),
    "used_mics": sol.used_mics,
})

__version__ = "0.1.0"
