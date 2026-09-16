# Figure titles

Written by `profiling/analyze.py`. The PNGs deliberately carry no title, so paste these in as captions.

### Scan time vs array size

Log-log. Each implementation is shown at its fastest block size.

![Scan time vs array size](img/perf/scan_vs_n.png)

### Stream compaction time vs array size

Log-log. Each implementation is shown at its fastest block size.

![Stream compaction time vs array size](img/perf/compact_vs_n.png)

### Scan time vs block size at n = 2^24

The ring marks the fastest block size for each implementation.

![Scan time vs block size at n = 2^24](img/perf/block_size_scan.png)

### Stream compaction time vs block size at n = 2^24

The ring marks the fastest block size for each implementation.

![Stream compaction time vs block size at n = 2^24](img/perf/block_size_compact.png)
