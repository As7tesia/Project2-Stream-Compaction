CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Yichen Huang
    - [LinkedIn](https://www.linkedin.com/in/yichen-huang-970b582bb/), [personal website](https://as7tesia.com/)
* Tested on: Windows 11, AMD Ryzen 5950X @ 4.3GHz (PBO Enabled), 64GB(3200 MT/s), RTX 3090 24GB


## Project Description
In this project I implemented:
- CPU Scan, CPU Stream Compaction
- Naive GPU Scan, Stupid Work-Efficient GPU Scan, Less-Stupid Work-Efficient GPU Scan
- GPU Stream Compaction

## Performance Analysis


## Why is my work-efficient scan so slow
### Observation
The work-efficient scan only beats naive past 2^22 and is still ~8x behind Thrust at 2^26. The block size
sweep is the real tell though: at 2^24 it goes 15.2 ms at block 32, 7.8 at 64, 4.6 at 128, 3.1 at 512. Naive
over the same range is flat (4.4 -> 4.3 ms). A memory-bound kernel doesn't halve when you double the block size.
That curve is the cost of dispatching blocks, so most of what I'm launching must not be doing anything.

### Reasoning

- I launch the same grid at every level: `paddedN / blockSize` blocks, 2^15 blocks (2^19 warps) at 2^24 with
  block 512. But the number of threads that pass the `index % 2^(d+1) == 0` check halves every level.
- Levels 0-4 (stride 2..32): every warp has some active lane, but only 16, 8, 4, 2, 1 of 32 lanes do work.
  The rest are masked, not free.
- Level 5 and up: the stride is wider than a warp, so only 1 in 2^(d-4) warps has an active lane at all, and
  that warp has exactly one. Every other warp computes an index, fails the check, and exits.
- Rough count for one sweep. Level d does `2^24 / 2^(d+1)` adds, so total adds are
  `2^23 + 2^22 + ... + 1 = 2^24 - 1`. At 32 lanes per warp that's 2^24 / 32 = 2^19 warps of actual work.
  I launch 24 levels x 2^19 = 24 * 2^19 warps. About 4% of what gets scheduled does anything.
- Kernel launch overhead is a footnote: 48 launches at a few us each is ~0.2 ms of the 3 ms.

- I also thought that padding to a power of two was a big issue because worst case it's almost 2x (33 -> 64), but I later realized in my sweep everything is already a power of 2.

### Fix
By doing some index math: at level d there are exactly `paddedN >> (d+1)` working threads, so launch that many (grid
size computed per level, min 1 block) and have thread `k` handle the node the old code reached at
`index = k << (d+1)`. The read/write offsets stay the same, the modulo check goes away. Same change in the
down-sweep.

Memory pattern doesn't change (the working lanes were already 2^(d+1) apart, I just stopped launching the
ones in between). Warps per sweep drop from 24 * 2^19 to ~2 * 2^19. Divergence only remains in the last 5
levels where there are fewer than 32 threads total, which you can't avoid.

**Results**

Both versions are in the sweep (`efficient` = compacted launch, `efficient_slow` = the original)
Scan at 2^24 by block size (ms):

| block | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| full launch | 16.26 | 8.33 | 4.60 | 3.17 | 2.98 | 4.05 |
| compacted | 2.02 | 2.07 | 2.19 | 2.13 | 2.14 | 2.12 |

The block size curve went flat, which is what you'd expect if the old curve was really measuring how
many dead blocks I was dispatching. Old version varied 8x across block sizes, new one varies 8%.

Scan at each version's best block (ms):

| n | full launch | compacted | speedup |
|---|---|---|---|
| 2^20 | 0.371 | 0.297 | 1.25x |
| 2^22 | 0.888 | 0.671 | 1.32x |
| 2^24 | 2.982 | 2.025 | 1.47x |
| 2^26 | 11.882 | 7.505 | 1.58x |

Compacted scan beats naive from 2^19 up and CPU from 2^20 up. Still ~4.8x behind Thrust at 2^26.
Below ~2^15 the two are indistinguishable. At that
size the old version only launches a handful of blocks per level anyway.
Speedup grows with n because the dead-block count scales with n while the launch floor doesn't.

![Scan time vs block size at n = 2^24](img/perf/block_size_scan.png)

*Scan time vs block size at n = 2^24. The ring marks the fastest block size for each implementation.*

![Scan time vs array size](img/perf/scan_vs_n.png)

*Scan time vs array size, log-log. Each implementation at its fastest block size.*


### Build notes

added `/Zc:preprocessor` to CMakeLists for MSVC builds
(and `-Xcompiler=/Zc:preprocessor` for the CUDA side). CUDA 13.3's Thrust/CCCL headers refuse to compile under
MSVC's traditional preprocessor and fail with a fatal `C1189` in `thrust.cu` without it.

added a second executable target, `cis5650_stream_compaction_bench` (`src/bench.cpp`), to the root CMakeLists.
It links the same `stream_compaction` library and only exists for `profiling/`

block size is a runtime parameter for the GPU implementations: `StreamCompaction::Naive::setBlockSize(int)` /
`getBlockSize()` and the same pair on `Efficient`, which lets the sweep cover 32..1024 without rebuilding.

running the sweep: `python profiling/run_sweep.py` writes a raw CSV under `profiling/raw/`, then
`python profiling/analyze.py` turns everything in there into `profiling/summary.{csv,md}` and the plots in
`img/perf/`.
