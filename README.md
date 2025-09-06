# acoustic-triangulation
A module for estimating the **direction (azimuth)** of a sound source from TDOA.

### Quick Start
```bash
python cli.py <<EOF
{
  "mics": {"A":[0,0], "B":[0.3,0]},
  "cfg": {"speed_of_sound":343.0, "ref_mic_id":"A"},
  "tdoa_s": {"B": 0.0004}
}
EOF
