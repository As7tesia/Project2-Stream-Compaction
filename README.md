CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)


### Part 5: why is my work-efficient scan slow

**What I noticed**

The work-efficient scan only beats naive past 2^22 and is still ~8x behind Thrust at 2^26. The block size
sweep is the real tell though: at 2^24 it goes 15.2 ms at block 32, 7.8 at 64, 4.6 at 128, 3.1 at 512. Naive
over the same range is flat (4.4 -> 4.3 ms). A memory-bound kernel doesn't halve when you double the block size.
That curve is the cost of dispatching blocks, so most of what I'm launching must not be doing anything.

**The problem**

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

**Thing I thought was the cause but isn't**

Padding to a power of two. Worst case it's almost 2x (33 -> 64) so it does matter in general, but every size
in my sweep is already a power of two, so it contributes zero to these numbers.

**The fix**

Index math only. At level d there are exactly `paddedN >> (d+1)` working threads, so launch that many (grid
size computed per level, min 1 block) and have thread `k` handle the node the old code reached at
`index = k << (d+1)`. The read/write offsets stay the same, the modulo check goes away. Same change in the
down-sweep.

Memory pattern doesn't change (the working lanes were already 2^(d+1) apart, I just stopped launching the
ones in between). Warps per sweep drop from 24 * 2^19 to ~2 * 2^19. Divergence only remains in the last 5
levels where there are fewer than 32 threads total, which you can't avoid.

**Results**

TODO after re-sweep: before/after table at 2^24, plot from `img/perf/`, new tuned block size.


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
