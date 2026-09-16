#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient_slow.h"

namespace StreamCompaction {
    namespace EfficientSlow {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        static int blockSize = 512;   // fastest at n = 2^24 in profiling/summary.md (pre-Part-5 sweep)

        void setBlockSize(int newBlockSize) {
            if (newBlockSize < 1 || newBlockSize > 1024) {
                throw std::invalid_argument("block size must be in 1..1024");
            }
            blockSize = newBlockSize;
        }

        int getBlockSize() {
            return blockSize;
        }
__global__ void kernScanUp(int n, int* data, int d)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n) return;
    if (!(index & (1 << (d+1)) - 1)) {     // = index mod (2^[d+1])
        data[index + (1 << (d+1)) - 1] += data[index + (1 << (d)) - 1];
    }
}

__global__ void kernScanDown(int n, int* data, int d) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n) return;
    if (!(index & (1 << (d+1)) - 1)) {     // = index mod (2^[d+1])
        int t = data[index + (1 << (d)) - 1];
        data[index + (1 << (d)) - 1] = data[index + (1 << (d+1)) - 1];
        data[index + (1 << (d+1)) - 1] += t;
    }
}
        /**
         * Exclusive scan of a device buffer, in place. No timer, no allocation.
         * d_data must be paddedN = 2^logn ints long with zeros past the real data.
         */
        void scanDevice(int paddedN, int logn, int *d_data) {
            int gridSize;
            // Up sweep
            for (int d = 0; d <= logn - 1; ++d) {
                gridSize = (paddedN + (blockSize - 1)) / blockSize;
                kernScanUp<<<gridSize, blockSize>>>(paddedN, d_data, d);
                checkCUDAError("kernScanUp failed");    
            }
            // x[n-1] = 0
            cudaMemset(d_data + paddedN - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset root failed");
            // Down sweep
            for (int d = logn - 1; d >= 0; --d) {
                kernScanDown<<<gridSize, blockSize>>>(paddedN, d_data, d);
                checkCUDAError("kernScanDown failed");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *d_data;
            int logn = ilog2ceil(n);
            int paddedN = 1 << logn;   // need size to be power of 2
            size_t size_bytes_padded = paddedN * sizeof(int);
            size_t size_bytes = n * sizeof(int);

            cudaMalloc((void**)&d_data, size_bytes_padded);
            checkCUDAError("cudaMalloc d_data failed");

            cudaMemcpy(d_data, idata, size_bytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> d_data failed");
            cudaMemset(d_data + n, 0, size_bytes_padded - size_bytes);
            checkCUDAError("cudaMemset padding failed");

            timer().startGpuTimer();
            scanDevice(paddedN, logn, d_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, d_data, size_bytes, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy d_data -> odata failed");
            cudaFree(d_data);
            checkCUDAError("cudaFree d_data failed");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int *dev_data, *dev_final, *dev_flag, *dev_indices;
            int paddedN = 1 << ilog2ceil(n);   // need size to be power of 2
            size_t size_bytes = sizeof(int) * n;
            size_t size_bytes_padded = paddedN * sizeof(int);
            int gridSize_n = (n + (blockSize - 1)) / blockSize;

            int count;

            cudaMalloc((void**)&dev_data, size_bytes);
            checkCUDAError("cudaMalloc dev_data failed");
            cudaMalloc((void**)&dev_final, size_bytes);
            checkCUDAError("cudaMalloc dev_final failed");
            cudaMalloc((void**)&dev_flag, size_bytes);
            checkCUDAError("cudaMalloc dev_flag failed");
            cudaMalloc((void**)&dev_indices, size_bytes_padded);
            checkCUDAError("cudaMalloc dev_indices failed");

            // init values in the padded parts so that it is identity when scanned
            cudaMemset(dev_indices + n, 0, sizeof(int) * (paddedN - n));
            checkCUDAError("cudaMemset dev_indices padding failed");
            cudaMemcpy(dev_data, idata, size_bytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> dev_data failed");

            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean<<<gridSize_n, blockSize>>>(n, dev_flag, dev_data);
            checkCUDAError("kernMapToBoolean failed");
            
            cudaMemcpy(dev_indices, dev_flag, size_bytes, cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy dev_flag -> dev_indices failed");
            scanDevice(paddedN, ilog2ceil(n), dev_indices);    // indices now contains scan result
            Common::kernScatter<<<gridSize_n, blockSize>>>(n, dev_final, dev_data, dev_flag, dev_indices);
            checkCUDAError("kernScatter failed");
            timer().endGpuTimer();
            
            cudaMemcpy(odata, dev_final, size_bytes, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_final -> odata failed");
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy count failed");

            cudaFree(dev_data);
            cudaFree(dev_final);
            cudaFree(dev_flag);
            cudaFree(dev_indices);
            checkCUDAError("cudaFree failed");

            return count + (idata[n-1] != 0);
        }
    }
}
