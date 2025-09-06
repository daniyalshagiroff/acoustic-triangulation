import sys, json
from triangulation.contracts import Mic, ArrayConfig, Observation
from triangulation.solver import estimate_azimuth

def main():
    data = json.load(sys.stdin)
    # ожидаем формат:
    # {
    #   "mics": {"A":[0,0], "B":[0.3,0]},  # метры
    #   "cfg": {"speed_of_sound":343.0, "ref_mic_id":"A"},
    #   "tdoa_s": {"B": 0.0004}            # секунды (B позже A на 0.4 мс)
    # }
    mics = {k: Mic(id=k, xy=tuple(v)) for k,v in data["mics"].items()}
    cfg  = ArrayConfig(**data.get("cfg", {}))
    obs  = Observation(tdoa_s=data["tdoa_s"])
    sol  = estimate_azimuth(mics, cfg, obs)
    print(json.dumps(sol.__dict__, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()