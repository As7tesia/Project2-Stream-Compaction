#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        /**
         * CUDA block size used by this module's kernels. 1..1024; anything else
         * throws std::invalid_argument. Lets the benchmark sweep block sizes
         * without a rebuild.
         */
        void setBlockSize(int blockSize);

        int getBlockSize();

        void scan(int n, int *odata, const int *idata);
    }
}
