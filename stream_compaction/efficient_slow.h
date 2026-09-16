#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace EfficientSlow {
        // Frozen baseline for Part 5: the work-efficient scan as written from the
        // slides, launching paddedN threads at every sweep level. Do not optimize.
        // Efficient:: is the upgraded version; this only exists for the A/B sweep.
        StreamCompaction::Common::PerformanceTimer& timer();

        /**
         * CUDA block size used by this module's kernels. 1..1024; anything else
         * throws std::invalid_argument. Lets the benchmark sweep block sizes
         * without a rebuild.
         */
        void setBlockSize(int blockSize);

        int getBlockSize();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
