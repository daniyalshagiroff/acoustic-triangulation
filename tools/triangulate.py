import argparse
import json
from typing import Dict

from triangulation.contracts import Mic, ArrayConfig, Observation
from triangulation.solver import estimate_azimuth


def parse_mics(s: str) -> Dict[str, Mic]:
    # format: "A:0,0 B:0.3,0 C:0,0.3"
    out: Dict[str, Mic] = {}
    for token in s.strip().split():
        mic_id, coords = token.split(":", 1)
        x_str, y_str = coords.split(",", 1)
        out[mic_id] = Mic(id=mic_id, xy=(float(x_str), float(y_str)))
    return out


def parse_tdoa(s: str) -> Dict[str, float]:
    # format: "B=+0.00040,C=-0.00020"
    out: Dict[str, float] = {}
    for pair in s.split(","):
        k, v = pair.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate azimuth from TDOA and mic geometry")
    ap.add_argument("--mics", required=True, help="e.g. 'A:0,0 B:0.3,0 C:0,0.3'")
    ap.add_argument("--ref", default="A", help="reference mic id (default: A)")
    ap.add_argument("--tdoa", required=True, help="e.g. 'B=+0.00040,C=-0.00020'")
    ap.add_argument("--c", type=float, default=343.0, help="speed of sound m/s (default 343.0)")
    args = ap.parse_args()

    mics = parse_mics(args.mics)
    cfg = ArrayConfig(speed_of_sound=args.c, ref_mic_id=args.ref)
    obs = Observation(tdoa_s=parse_tdoa(args.tdoa))

    sol = estimate_azimuth(mics, cfg, obs)
    print(json.dumps({
        "azimuth_deg": sol.azimuth_deg,
        "quality": sol.quality,
        "used_mics": sol.used_mics,
    }))


if __name__ == "__main__":
    main()


