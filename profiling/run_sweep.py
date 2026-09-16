#!/usr/bin/env python
"""Drive cis5650_stream_compaction_bench over the (implementation, block size, n) grid
and collect every timed run into one raw CSV.

    python profiling/run_sweep.py                       # full sweep, 2^8..2^26
    python profiling/run_sweep.py --n 8..12 --repeats 2 # quick smoke test
    python profiling/run_sweep.py --dry-run             # just print the commands

One subprocess per (impl, op, block); n and the repeats loop inside the process, so CUDA
startup is paid ~20 times for the whole sweep and a group that dies (oversized block, OOM)
only costs that group. Rows already flushed are kept either way.

Writes profiling/raw/<YYYYmmdd-HHMMSS>[_tag].csv: one `#` metadata line, the CSV header,
then the bench rows verbatim. Exits non-zero if any group failed.
"""
import argparse
import io
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_EXE = os.path.join(ROOT, "build", "bin", "Release", "cis5650_stream_compaction_bench.exe")
CSV_HEADER = "impl,op,n,block_size,run,ms,ok"

# Mirrors the table in src/bench.cpp. The flag says whether block size is a knob.
BENCHMARKS = [
    ("cpu", "scan", False),
    ("cpu", "compact", False),
    ("cpu_scan", "compact", False),
    ("naive", "scan", True),
    ("efficient", "scan", True),
    ("efficient", "compact", True),
    ("efficient_slow", "scan", True),
    ("efficient_slow", "compact", True),
    ("thrust", "scan", False),
]


# ---------------------------------------------------------------- environment

def query(cmd, default=""):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return default
    return out.stdout.strip() if out.returncode == 0 else default


def gpu_check(skip):
    """Return (name, driver). Abort if something else is already using the GPU."""
    line = query(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu",
                  "--format=csv,noheader,nounits"])
    if not line:
        print("nvidia-smi not available; skipping the idle check")
        return "unknown", "unknown"
    name, driver, util = [f.strip() for f in line.splitlines()[0].split(",")]
    if int(util) > 10:
        msg = (f"GPU is at {util}% utilization before the sweep ({name}). "
               "Close other GPU windows, or pass --no-gpu-check.")
        if not skip:
            raise SystemExit("abort: " + msg)
        print("warning: " + msg)
    else:
        print(f"GPU idle check: {name}, driver {driver}, {util}% utilization")
    return name, driver


def git_sha():
    sha = query(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], "unknown")
    if query(["git", "-C", ROOT, "status", "--porcelain"]):
        sha += "+dirty"
    return sha


# ---------------------------------------------------------------- the sweep

def groups(args):
    """(impl, op, block) tuples to run; block is None where it does not apply."""
    impls = None if args.impl == "all" else args.impl.split(",")
    ops = None if args.op == "all" else args.op.split(",")
    known_impls = {b[0] for b in BENCHMARKS}
    known_ops = {b[1] for b in BENCHMARKS}
    for name in impls or []:
        if name not in known_impls:
            raise SystemExit(f"abort: unknown --impl '{name}', expected one of {sorted(known_impls)}")
    for name in ops or []:
        if name not in known_ops:
            raise SystemExit(f"abort: unknown --op '{name}', expected one of {sorted(known_ops)}")

    blocks = [int(b) for b in args.block.split(",")]
    out = []
    for impl, op, tunable in BENCHMARKS:
        if (impls and impl not in impls) or (ops and op not in ops):
            continue
        out += [(impl, op, b) for b in blocks] if tunable else [(impl, op, None)]
    if not out:
        raise SystemExit(f"abort: no benchmark matches --impl {args.impl} --op {args.op}")
    return out


def command(args, impl, op, block):
    cmd = [args.exe, "--impl", impl, "--op", op, "--n", args.n,
           "--repeats", str(args.repeats), "--warmup", str(args.warmup),
           "--maxval", str(args.maxval), "--seed", str(args.seed), "--no-header"]
    if block is not None:
        cmd += ["--block", str(block)]
    if not args.no_verify:
        cmd.append("--verify")
    return cmd


def report(impl, op, block, n, samples, bad):
    flag = "  <-- FAILED VERIFY" if bad else ""
    print(f"{impl:<10} {op:<8} {'-' if block is None else block:>6} {n:>10} {len(samples):>5} "
          f"{statistics.median(samples):>11.4f} {min(samples):>11.4f}{flag}", flush=True)


def run_group(args, impl, op, block, sink):
    """Run one subprocess, stream its rows into `sink`, print a line per n.

    Returns (rows_written, error_message_or_None, stderr_text).
    """
    cmd = command(args, impl, op, block)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1, cwd=ROOT)
    stderr = []
    drain = threading.Thread(target=lambda: stderr.append(proc.stderr.read()), daemon=True)
    drain.start()
    timed_out = threading.Event()

    def on_timeout():
        timed_out.set()
        proc.kill()

    watchdog = threading.Timer(args.timeout, on_timeout)
    watchdog.start()

    written, current_n, samples, bad = 0, None, [], False
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 7:
                stderr.append(f"unparseable row: {line}\n")
                continue
            sink.write(line + "\n")
            sink.flush()
            written += 1
            n, ms, ok = int(fields[2]), float(fields[5]), int(fields[6])
            if n != current_n:
                if samples:
                    report(impl, op, block, current_n, samples, bad)
                current_n, samples, bad = n, [], False
            samples.append(ms)
            bad = bad or ok == 0
        if samples:
            report(impl, op, block, current_n, samples, bad)
    finally:
        proc.wait()
        watchdog.cancel()
        drain.join(timeout=5)

    errtext = "".join(chunk for chunk in stderr if chunk)
    if timed_out.is_set():
        return written, f"timed out after {args.timeout}s", errtext
    if proc.returncode != 0:
        return written, f"exit code {proc.returncode}", errtext
    if written == 0:
        return written, "produced no rows", errtext
    return written, None, errtext


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default=DEFAULT_EXE, help="benchmark executable (Release build)")
    ap.add_argument("--impl", default="all", help="all, or a comma list of implementations")
    ap.add_argument("--op", default="all", help="all, or a comma list of scan,compact")
    ap.add_argument("--n", default="8..26", help="sizes; values <= 31 are log2 exponents")
    ap.add_argument("--block", default="32,64,128,256,512,1024", help="CUDA block sizes")
    ap.add_argument("--repeats", type=int, default=5, help="timed runs per configuration")
    ap.add_argument("--warmup", type=int, default=1, help="untimed runs before them")
    ap.add_argument("--maxval", type=int, default=4, help="input values are uniform in [0, V)")
    ap.add_argument("--seed", type=int, default=1234, help="input generator seed")
    ap.add_argument("--no-verify", action="store_true", help="skip the correctness check")
    ap.add_argument("--tag", default="", help="suffix for the raw CSV filename")
    ap.add_argument("--timeout", type=float, default=900, help="seconds before a group is killed")
    ap.add_argument("--raw-dir", default=os.path.join(HERE, "raw"))
    ap.add_argument("--no-gpu-check", action="store_true", help="run even if the GPU is busy")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = ap.parse_args()

    plan = groups(args)
    if args.dry_run:
        for impl, op, block in plan:
            print(subprocess.list2cmdline(command(args, impl, op, block)))
        print(f"\n{len(plan)} group(s)")
        return 0
    if not os.path.isfile(args.exe):
        raise SystemExit(f"abort: executable not found: {args.exe}\n"
                         "       cmake --build build --config Release")

    gpu, driver = gpu_check(args.no_gpu_check)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.raw_dir, exist_ok=True)
    out_path = os.path.join(args.raw_dir, stamp + ".csv")
    log_path = os.path.join(args.raw_dir, stamp + ".log")

    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu,
        "driver": driver,
        "git": git_sha(),
        "args": subprocess.list2cmdline(sys.argv[1:]) or "(defaults)",
    }
    print(f"\nwriting {os.path.relpath(out_path, ROOT)}  ({len(plan)} groups)\n")
    print(f"{'impl':<10} {'op':<8} {'block':>6} {'n':>10} {'runs':>5} {'median ms':>11} {'min ms':>11}")

    failures, total_rows, started = [], 0, time.time()
    with io.open(out_path, "w", encoding="utf-8", newline="\n") as sink:
        sink.write("# " + ",".join(f"{k}={str(v).replace(',', ';')}" for k, v in meta.items()) + "\n")
        sink.write(CSV_HEADER + "\n")
        sink.flush()
        for impl, op, block in plan:
            try:
                written, error, errtext = run_group(args, impl, op, block, sink)
            except KeyboardInterrupt:
                print("\ninterrupted; partial results kept in "
                      f"{os.path.relpath(out_path, ROOT)}")
                return 130
            total_rows += written
            if error:
                label = f"{impl}/{op} block={block}"
                failures.append((f"{label}: {error} ({written} rows)", errtext))
                print(f"  !! {label}: {error}", flush=True)

    print(f"\n{time.time() - started:.1f}s total")
    if failures:
        with io.open(log_path, "w", encoding="utf-8", newline="\n") as log:
            for headline, errtext in failures:
                log.write(f"=== {headline}\n{errtext.rstrip()}\n\n")
        print(f"{len(failures)} group(s) failed, stderr in {os.path.relpath(log_path, ROOT)}:")
        for headline, _ in failures:
            print("  " + headline)
        return 1
    print(f"{total_rows} rows -> {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
