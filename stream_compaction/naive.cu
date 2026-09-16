#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        static int blockSize = 512;   // fastest at n = 2^24 in profiling/summary.md

        void setBlockSize(int newBlockSize) {
            if (newBlockSize < 1 || newBlockSize > 1024) {
                throw std::invalid_argument("block size must be in 1..1024");
            }
            blockSize = newBlockSize;
        }

        int getBlockSize() {
            return blockSize;
        }
        // TODO: __global__
        __global__ void kernScan(int n, int *d_data_r, int *d_data_w, int d) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;
            if (index >= (1 << (d - 1))) { 
                d_data_w[index] = d_data_r[index - (1 << (d - 1))] + d_data_r[index];
            } else {
                d_data_w[index] = d_data_r[index];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int gridSize = (n + (blockSize- 1)) / blockSize;
            int logn = ilog2ceil(n);
            int *d_data_r, *d_data_w;
            size_t size_bytes = sizeof(int) * n;
            
            cudaMalloc((void**)&d_data_r, size_bytes);
            checkCUDAError("cudaMalloc failed for d_data_r failed");
            
            cudaMalloc((void**)&d_data_w, size_bytes);
            checkCUDAError("cudaMalloc failed for d_data_w failed");
            
            // d_data = idata
            cudaMemcpy(d_data_r, idata, size_bytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed for d_data_r failed");
            
            cudaMemcpy(d_data_w, idata, size_bytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed for d_data_r failed");
            
            timer().startGpuTimer();
            // TODO: launch kernel
            for (int d = 1; d <= logn; ++d) {
                kernScan<<<gridSize, blockSize>>>(n, d_data_r, d_data_w, d);
                std::swap(d_data_r, d_data_w);
            }
            checkCUDAError("kernScan failed");
            
            timer().endGpuTimer();
            
            cudaMemcpy(odata + 1, d_data_r, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy failed for odata failed");
            odata[0] = 0;

            cudaFree(d_data_r);
            checkCUDAError("cudaFree failed on d_data_r");
            cudaFree(d_data_w);
            checkCUDAError("cudaFree failed on d_data_w");
        }
    }
}
