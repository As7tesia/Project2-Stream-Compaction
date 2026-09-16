#!/usr/bin/env python
"""Summarize profiling/raw/*.csv (written by run_sweep.py) and draw the README graphs.

    python profiling/analyze.py [--raw "profiling/raw/*.csv"] [--images img/perf] [--block-n 24]

Every raw file is one sweep. Rows that failed verification (ok == 0) are dropped, and
where several sweeps cover the same (impl, op, n, block_size) the newest file wins, so
re-running a subset of the grid just overrides those points.

Outputs: profiling/summary.csv, profiling/summary.md, and four dark-themed PNGs in
--images (scan_vs_n, compact_vs_n, block_size_scan, block_size_compact).
"""
import argparse
import glob
import io
import os
import time

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEY = ["impl", "op", "n", "block_size"]

# Dark surface. `dark_background` on its own is pure black, which blooms on an OLED and
# looks nothing like GitHub's dark theme, so the greys below are applied over it.
FIGURE, SURFACE = "#17191c", "#1e2125"
INK, INK2, MUTED, GRID, AXIS = "#e8eaed", "#b0b4ba", "#8a8f96", "#33373d", "#4a4f57"

# Okabe-Ito, brightened where it needs to carry on a dark surface. CPU is the neutral
# baseline; the three GPU implementations get the three most separable hues.
IMPL_ORDER = ["cpu", "cpu_scan", "naive", "efficient", "thrust"]
IMPL_COLOR = {
    "cpu": "#c7cad1",
    "cpu_scan": "#f0e442",
    "naive": "#56b4e9",
    "efficient": "#e69f00",
    "thrust": "#31c99e",
}
IMPL_NAME = {
    "cpu": "CPU",
    "cpu_scan": "CPU scan + scatter",
    "naive": "Naive GPU",
    "efficient": "Work-efficient GPU",
    "thrust": "Thrust",
}
OP_NAME = {"scan": "Scan", "compact": "Stream compaction"}

# (path, title, caption) for every figure drawn this run; written out as titles.md.
FIGURES = []


# ---------------------------------------------------------------- loading

def load_all(pattern):
    paths = sorted(glob.glob(pattern), key=lambda p: (os.path.getmtime(p), p))
    if not paths:
        raise SystemExit(f"no raw CSVs match {pattern}; run profiling/run_sweep.py first")
    frames = []
    for rank, path in enumerate(paths):
        d = pd.read_csv(path, comment="#")
        missing = set(KEY + ["run", "ms", "ok"]) - set(d.columns)
        if missing:
            print(f"skip {os.path.basename(path)}: missing columns {sorted(missing)}")
            continue
        d["file_rank"] = rank
        frames.append(d)
    if not frames:
        raise SystemExit(f"no usable CSVs match {pattern}")
    df = pd.concat(frames, ignore_index=True)

    bad = int((df.ok == 0).sum())
    if bad:
        print(f"dropping {bad} row(s) that failed verification")
    df = df[df.ok == 1]
    # A later sweep supersedes an earlier one point by point.
    df = df[df.file_rank == df.groupby(KEY)["file_rank"].transform("max")]
    if df.empty:
        raise SystemExit("every row was dropped; nothing to summarize")
    return df.reset_index(drop=True), len(frames)


def summarize(df):
    s = (df.groupby(KEY)["ms"]
           .agg(runs="count", median_ms="median", min_ms="min", mean_ms="mean", std_ms="std")
           .reset_index())
    return s.fillna({"std_ms": 0.0}).sort_values(KEY).reset_index(drop=True)


def best_blocks(summary, ref_n):
    """Block size with the lowest median at ref_n (or the largest n present) per impl/op."""
    out = {}
    for (impl, op), d in summary.groupby(["impl", "op"]):
        at = d[d.n == ref_n]
        if at.empty:
            at = d[d.n == d.n.max()]
        out[(impl, op)] = int(at.loc[at.median_ms.idxmin(), "block_size"])
    return out


# ---------------------------------------------------------------- tables

def fmt_n(n):
    n = int(n)
    k = n.bit_length() - 1
    if n == 1 << k:
        return f"2^{k}"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def fmt_n_math(n):
    """Same label, as mathtext, so plots get a real superscript instead of "2^26"."""
    n = int(n)
    k = n.bit_length() - 1
    return f"$2^{{{k}}}$" if n == 1 << k else fmt_n(n)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def median_at(summary, impl, op, block, n):
    d = summary[(summary.impl == impl) & (summary.op == op)
                & (summary.block_size == block) & (summary.n == n)]
    return None if d.empty else float(d.iloc[0].median_ms)


def block_table(summary, ref_n):
    d = summary[(summary.n == ref_n) & (summary.block_size > 0)]
    if d.empty or d.block_size.nunique() < 2:
        return None
    pairs = [(impl, op) for impl in IMPL_ORDER for op in ("scan", "compact")
             if ((d.impl == impl) & (d.op == op)).any()]
    header = ["Block size"] + [f"{IMPL_NAME[i]} {op} ms" for i, op in pairs]
    rows = []
    for block in sorted(d.block_size.unique()):
        row = [str(int(block))]
        for impl, op in pairs:
            ms = median_at(summary, impl, op, block, ref_n)
            row.append("" if ms is None else f"{ms:.4f}")
        rows.append(row)
    return md_table(header, rows)


def vs_n_table(summary, op, best, impls):
    d = summary[summary.op == op]
    impls = [i for i in impls if (d.impl == i).any()]
    if not impls:
        return None
    others = [i for i in impls if i != "cpu"]
    header = ["n"] + [f"{IMPL_NAME[i]} ms" for i in impls] + [f"{IMPL_NAME[i]} vs CPU" for i in others]
    rows = []
    for n in sorted(d.n.unique()):
        times = {i: median_at(summary, i, op, best[(i, op)], n) for i in impls}
        row = [fmt_n(n)] + ["" if times[i] is None else f"{times[i]:.4f}" for i in impls]
        for i in others:
            base, got = times.get("cpu"), times[i]
            if not base or not got:
                row.append("")
                continue
            ratio = base / got
            row.append(f"{ratio:.2f}x" if ratio >= 0.1 else f"{ratio:.3f}x")
        rows.append(row)
    return md_table(header, rows)


def write_summary(summary, out_dir, ref_n, best, n_files, n_rows):
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False, float_format="%.6f")

    parts = [
        "# Stream compaction profiling summary\n",
        f"{n_rows} timed runs from {n_files} raw sweep file(s). Times are the median over the "
        "repeats of one configuration: CUDA events for the GPU implementations, `std::chrono` "
        "for the CPU ones. `cudaMalloc` and the host/device copies sit outside the timers, so "
        "these are kernel times only. `block_size` 0 means the implementation has no block "
        "size to tune.\n",
    ]
    t = block_table(summary, ref_n)
    if t:
        parts.append(f"## Block size sweep at n = {fmt_n(ref_n)}\n\n{t}\n")
    picked = ", ".join(f"{IMPL_NAME[i]} {op}: {b}" for (i, op), b in sorted(best.items()) if b)
    if picked:
        parts.append(f"Best block size at that size — {picked}. The tables below use it.\n")
    t = vs_n_table(summary, "scan", best, ["cpu", "naive", "efficient", "thrust"])
    if t:
        parts.append(f"## Scan vs array size\n\n{t}\n")
    t = vs_n_table(summary, "compact", best, ["cpu", "cpu_scan", "efficient"])
    if t:
        parts.append(f"## Stream compaction vs array size\n\n{t}\n")

    with io.open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------- plots

def style(ax, xlabel, ylabel):
    # No title is drawn into the figure on purpose; titles go to img/perf/titles.md
    # so they can be README captions instead of baked-in pixels.
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, which="both")
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def n_axis(ax, ns):
    # A full sweep is 19 sizes; labelling all of them runs the exponents together, so
    # label every k-th one and leave an unlabelled minor tick on the rest.
    ns = sorted({int(v) for v in ns})
    labelled = ns[::1 + len(ns) // 10]
    if ns[-1] not in labelled:
        labelled.append(ns[-1])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(labelled))
    ax.xaxis.set_minor_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_n_math(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plain_log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def legend(ax, ncols):
    # Under the axes: the lines cross most of the plot area on a log-log sweep, so any
    # in-axes corner ends up sitting on top of data.
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncols=min(ncols, 4))


def save(fig, path, title, caption=""):
    # Retry: on Windows a thumbnailer or file watcher sometimes has the previous PNG
    # memory-mapped for a moment, and opening it for truncation fails with EINVAL/EACCES.
    for attempt in range(20):
        try:
            fig.savefig(path, dpi=150, facecolor=FIGURE, bbox_inches="tight")
            break
        except OSError as e:
            if e.errno not in (13, 22) or attempt == 19:
                raise
            time.sleep(0.25)
    plt.close(fig)
    FIGURES.append((path, title, caption))
    print("wrote", os.path.relpath(path, ROOT))


def write_titles(out_dir):
    """Titles and ready-to-paste markdown for the figures, kept out of the PNGs."""
    if not FIGURES:
        return
    lines = ["# Figure titles\n",
             "Written by `profiling/analyze.py`. The PNGs deliberately carry no title, so "
             "paste these in as captions.\n"]
    for path, title, caption in FIGURES:
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        lines.append(f"### {title}\n")
        if caption:
            lines.append(caption + "\n")
        lines.append(f"![{title}]({rel})\n")
    path = os.path.join(out_dir, "titles.md")
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print("wrote", os.path.relpath(path, ROOT))


def plot_vs_n(summary, images, op, best, impls, fname):
    d = summary[summary.op == op]
    impls = [i for i in impls if (d.impl == i).any()]
    if not impls:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=FIGURE)
    for impl in impls:
        block = best[(impl, op)]
        s = d[(d.impl == impl) & (d.block_size == block)].sort_values("n")
        if s.empty:
            continue
        ax.plot(s.n, s.median_ms, "-", color=IMPL_COLOR[impl], lw=2, marker="o", ms=4,
                label=IMPL_NAME[impl] + (f", block {block}" if block else ""))
    style(ax, "Array size n", "Time, ms (median)")
    n_axis(ax, d.n.unique())
    plain_log_y(ax)
    legend(ax, len(impls))
    save(fig, os.path.join(images, fname), f"{OP_NAME[op]} time vs array size",
         "Log-log. Each implementation is shown at its fastest block size.")


def plot_block_size(summary, images, op, ref_n, fname):
    d = summary[(summary.op == op) & (summary.n == ref_n) & (summary.block_size > 0)]
    if d.empty or d.block_size.nunique() < 2:
        print(f"skip {fname}: need at least two block sizes at n = {fmt_n(ref_n)}")
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=FIGURE)
    for impl in IMPL_ORDER:
        s = d[d.impl == impl].sort_values("block_size")
        if s.empty:
            continue
        ax.plot(s.block_size, s.median_ms, "-", color=IMPL_COLOR[impl], lw=2, marker="o", ms=5,
                label=IMPL_NAME[impl])
        low = s.loc[s.median_ms.idxmin()]
        ax.plot(low.block_size, low.median_ms, "o", ms=12, mfc="none",
                mec=IMPL_COLOR[impl], mew=1.5)
    style(ax, "Block size (threads)", "Time, ms (median)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(sorted(d.block_size.unique())))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    legend(ax, int(d.impl.nunique()))
    save(fig, os.path.join(images, fname),
         f"{OP_NAME[op]} time vs block size at n = {fmt_n(ref_n)}",
         "The ring marks the fastest block size for each implementation.")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(HERE, "raw", "*.csv"))
    ap.add_argument("--images", default=os.path.join(ROOT, "img", "perf"))
    ap.add_argument("--block-n", type=int, default=24,
                    help="array size for the block-size tables and plots; <= 31 is an exponent")
    args = ap.parse_args()

    plt.style.use("dark_background")
    df, n_files = load_all(args.raw)
    summary = summarize(df)

    ref_n = 1 << args.block_n if args.block_n <= 31 else args.block_n
    if ref_n not in set(summary.n):
        ref_n = int(summary.n.max())
        print(f"n = {fmt_n(1 << args.block_n)} is not in the data; using {fmt_n(ref_n)} instead")
    best = best_blocks(summary, ref_n)

    os.makedirs(args.images, exist_ok=True)
    write_summary(summary, HERE, ref_n, best, n_files, len(df))
    print(f"{len(summary)} configurations -> profiling/summary.csv, profiling/summary.md")

    plot_vs_n(summary, args.images, "scan", best,
              ["cpu", "naive", "efficient", "thrust"], "scan_vs_n.png")
    plot_vs_n(summary, args.images, "compact", best,
              ["cpu", "cpu_scan", "efficient"], "compact_vs_n.png")
    plot_block_size(summary, args.images, "scan", ref_n, "block_size_scan.png")
    plot_block_size(summary, args.images, "compact", ref_n, "block_size_compact.png")
    write_titles(args.images)


if __name__ == "__main__":
    main()
