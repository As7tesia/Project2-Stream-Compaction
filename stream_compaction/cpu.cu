#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = idata[i-1] + odata[i-1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            int counter = 0;
            timer().startCpuTimer();
            // TODO
            for (int k = 0; k < n; ++k) {
                if (idata[k] != 0) {
                    odata[counter] = idata[k];
                    ++counter;
                }
            }
            timer().endCpuTimer();
            return counter;
        }

        // CPU scatter
        void scatter(int n, int *odata, const int *idata, int *flag, int *idx) {
            for (int k = 0; k < n; ++k) {
                if (flag[k] != 0) {
                    odata[idx[k]] = idata[k];
                }
            }
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *flag = new int[n];
            int *scan_result = new int[n];
            timer().startCpuTimer();

            // step 1 calculate temp array of flags
            for (int i = 0; i < n; ++i) {
                flag[i] = idata[i] != 0;
            }

            // step 2 scan the temp flag array
            scan_result[0] = 0;
            for (int i = 1; i < n; ++i) {
                scan_result[i] = flag[i-1] + scan_result[i-1];
            }

            // step 3 scatter final array
            scatter(n, odata, idata, flag, scan_result);

            timer().endCpuTimer();
            int elements_remain = scan_result[n - 1] + flag[n - 1];
            delete[] flag;
            delete[] scan_result;
            return elements_remain;
        }
    }
}
