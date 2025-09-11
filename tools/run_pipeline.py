import argparse
import json
import os
import re
import subprocess
import sys
from typing import Optional, Tuple


THIS_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))


def run(cmd: list[str]) -> Tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    # Ensure our project root is on PYTHONPATH for child scripts in tools/
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (PROJ_ROOT + (os.pathsep + existing_pp if existing_pp else ""))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=PROJ_ROOT,
    )
    out, err = proc.communicate()
    # ensure non-None strings
    out = out if out is not None else ""
    err = err if err is not None else ""
    return proc.returncode, out, err


def ensure_16k(wavs: list[str], out_dir: str, keep_name: bool = True, mono: bool = True) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(THIS_DIR, "resample_to_16k.py"), *wavs, "--out_dir", out_dir]
    if mono:
        cmd.append("--mono")
    if keep_name:
        cmd.append("--keep_name")
    code, out, err = run(cmd)
    if code != 0:
        print(err or out, file=sys.stderr)
        sys.exit(code)
    produced = []
    for p in wavs:
        base = os.path.splitext(os.path.basename(p))[0]
        out_name = f"{base}.wav" if keep_name else f"{base}_16k.wav"
        produced.append(os.path.join(out_dir, out_name))
    return produced


def maybe_generate_from_source(source_wav: Optional[str], out_dir: str) -> Optional[Tuple[str, str, str]]:
    if not source_wav:
        return None
    cmd = [sys.executable, os.path.join(THIS_DIR, "gen_from_source.py"), source_wav, "--out_dir", out_dir]
    code, out, err = run(cmd)
    if code != 0:
        print(err or out, file=sys.stderr)
        sys.exit(code)
    A = os.path.join(out_dir, "micA.wav")
    B = os.path.join(out_dir, "micB.wav")
    C = os.path.join(out_dir, "micC.wav")
    return (A, B, C)


def run_align_wavs(micA: str, micB: str, micC: str, out_npz: str,
                   frame_ms: float = 64.0, hop_ms: float = 32.0,
                   offsets_ms: Tuple[float, float, float] = (0.0, 120.0, -80.0), jitter_ms: float = 0.0,
                   fs: int = 16000) -> None:
    os.makedirs(os.path.dirname(out_npz) or THIS_DIR, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(THIS_DIR, "align_wavs.py"),
        micA, micB, micC,
        "--out_npz", out_npz,
        "--fs", str(fs),
        "--frame_ms", str(frame_ms),
        "--hop_ms", str(hop_ms),
        "--offsets_ms", str(offsets_ms[0]), str(offsets_ms[1]), str(offsets_ms[2]),
        "--jitter_ms", str(jitter_ms),
        "--n_print", "5",
    ]
    code, out, err = run(cmd)
    if code != 0:
        print(err or out, file=sys.stderr)
        sys.exit(code)
    print(out.strip())


def run_tdoa_from_npz(npz_path: str, max_tdoa_ms: float = 2.0, n_print: int = 10) -> Tuple[float, float]:
    cmd = [
        sys.executable, os.path.join(THIS_DIR, "tdoa.py"),
        "--from_npz", npz_path,
        "--max_tdoa_ms", str(max_tdoa_ms),
        "--print", str(n_print),
    ]
    code, out, err = run(cmd)
    if code != 0:
        print(err or out, file=sys.stderr)
        sys.exit(code)
    print(out.strip())
    mb = None
    mc = None
    for line in out.splitlines():
        m1 = re.search(r"TDOA B-A \(sec\): median=([+-]?[0-9]*\.?[0-9]+)", line)
        if m1:
            mb = float(m1.group(1))
        m2 = re.search(r"TDOA C-A \(sec\): median=([+-]?[0-9]*\.?[0-9]+)", line)
        if m2:
            mc = float(m2.group(1))
    if mb is None or mc is None:
        print("[!] Could not parse TDOA medians from output", file=sys.stderr)
        sys.exit(2)
    return mb, mc


def run_triangulate(mics: str, ref: str, tdoa_b: float, tdoa_c: float, c: float = 343.0) -> dict:
    tdoa_arg = f"B={tdoa_b:+.6f},C={tdoa_c:+.6f}"
    cmd = [
        sys.executable, os.path.join(THIS_DIR, "triangulate.py"),
        "--mics", mics,
        "--ref", ref,
        "--tdoa", tdoa_arg,
        "--c", str(c),
    ]
    code, out, err = run(cmd)
    if code != 0:
        print(err or out, file=sys.stderr)
        sys.exit(code)
    # Try to parse the last JSON-looking line
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and '"' in ln:
            try:
                return json.loads(ln)
            except Exception:
                continue
    # Fallback: try entire stdout
    try:
        return json.loads(out)
    except Exception:
        print(out)
        print("[!] triangulate output is not valid JSON", file=sys.stderr)
        sys.exit(3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full pipeline: optional gen -> align_wavs -> tdoa -> triangulate")
    ap.add_argument("--source", help="Path to single source WAV to synthesize 3 mics (optional)")
    ap.add_argument("--micA", help="Path to micA.wav (if --source not used)")
    ap.add_argument("--micB", help="Path to micB.wav (if --source not used)")
    ap.add_argument("--micC", help="Path to micC.wav (if --source not used)")
    ap.add_argument("--mics", default="A:0,0 B:0.3,0 C:0,0.3", help="Mic geometry, e.g. 'A:0,0 B:0.3,0 C:0,0.3'")
    ap.add_argument("--ref", default="A", help="Reference mic id (default: A)")
    ap.add_argument("--c", type=float, default=343.0, help="Speed of sound m/s (default: 343.0)")
    ap.add_argument("--work", default=os.path.join(THIS_DIR, "_work"), help="Working directory for outputs")
    ap.add_argument("--align_frame_ms", type=float, default=64.0)
    ap.add_argument("--align_hop_ms", type=float, default=32.0)
    ap.add_argument("--align_offsets_ms", nargs=3, type=float, default=[0.0, 120.0, -80.0])
    ap.add_argument("--align_jitter_ms", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)

    if args.source:
        print("[1/4] Generating micA/B/C from source...")
        triplet = maybe_generate_from_source(args.source, args.work)
        if triplet is None:
            print("[!] Failed to generate from source", file=sys.stderr)
            sys.exit(1)
        micA, micB, micC = triplet
    else:
        if not (args.micA and args.micB and args.micC):
            print("[!] Provide --source or all of --micA --micB --micC", file=sys.stderr)
            sys.exit(1)
        micA, micB, micC = args.micA, args.micB, args.micC

    print("[2/4] Ensuring 16 kHz mono copies...")
    micA16, micB16, micC16 = ensure_16k([micA, micB, micC], os.path.join(args.work, "16k"), keep_name=True, mono=True)

    print("[3/4] Aligning windows (slots)...")
    out_npz = os.path.join(args.work, "aligned_windows.npz")
    run_align_wavs(
        micA16, micB16, micC16, out_npz,
        frame_ms=args.align_frame_ms,
        hop_ms=args.align_hop_ms,
        offsets_ms=(args.align_offsets_ms[0], args.align_offsets_ms[1], args.align_offsets_ms[2]),
        jitter_ms=args.align_jitter_ms,
        fs=16000,
    )

    print("[4/4] Estimating TDOA from aligned windows...")
    tdoa_ba, tdoa_ca = run_tdoa_from_npz(out_npz, max_tdoa_ms=2.0, n_print=10)

    print("[=] Triangulating azimuth...")
    result = run_triangulate(args.mics, args.ref, tdoa_ba, tdoa_ca, c=args.c)
    final = {
        "mics": args.mics,
        "ref": args.ref,
        "tdoa_s": {"B": tdoa_ba, "C": tdoa_ca},
        "speed_of_sound": args.c,
        "solution": result,
        "artifacts": {
            "aligned_npz": out_npz,
            "micA_16k": micA16,
            "micB_16k": micB16,
            "micC_16k": micC16,
        }
    }
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
