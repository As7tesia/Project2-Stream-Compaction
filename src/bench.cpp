/**
 * @file      bench.cpp
 * @brief     Benchmark driver for the stream compaction implementations.
 *
 * Sweeps array size and CUDA block size without rebuilding, and writes one CSV
 * row per timed run to stdout:
 *
 *     impl,op,n,block_size,run,ms,ok
 *     efficient,scan,16777216,256,0,3.412000,1
 *
 * Everything else (progress, warnings, errors) goes to stderr, so the caller can
 * just redirect stdout into a file. Driven by profiling/run_sweep.py. The course
 * test harness in main.cpp is a separate executable and is untouched.
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <stream_compaction/cpu.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/efficient_slow.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/thrust.h>

namespace {

// ------------------------------------------------------------------ the table

struct Benchmark {
    const char *impl;
    const char *op;
    void (*setBlock)(int);                              // nullptr => block size is not a knob
    std::function<void(int, int *, const int *)> run;   // int-returning entry points wrapped
    std::function<float()> elapsedMs;                   // that module's timer
};

std::vector<Benchmark> makeTable() {
    using namespace StreamCompaction;
    auto cpuMs = [] { return CPU::timer().getCpuElapsedTimeForPreviousOperation(); };
    auto naiveMs = [] { return Naive::timer().getGpuElapsedTimeForPreviousOperation(); };
    auto efficientMs = [] { return Efficient::timer().getGpuElapsedTimeForPreviousOperation(); };
    auto efficientSlowMs = [] { return EfficientSlow::timer().getGpuElapsedTimeForPreviousOperation(); };
    auto thrustMs = [] { return Thrust::timer().getGpuElapsedTimeForPreviousOperation(); };
    return {
        {"cpu", "scan", nullptr,
         [](int n, int *o, const int *i) { CPU::scan(n, o, i); }, cpuMs},
        {"cpu", "compact", nullptr,
         [](int n, int *o, const int *i) { CPU::compactWithoutScan(n, o, i); }, cpuMs},
        {"cpu_scan", "compact", nullptr,
         [](int n, int *o, const int *i) { CPU::compactWithScan(n, o, i); }, cpuMs},
        {"naive", "scan", &Naive::setBlockSize,
         [](int n, int *o, const int *i) { Naive::scan(n, o, i); }, naiveMs},
        {"efficient", "scan", &Efficient::setBlockSize,
         [](int n, int *o, const int *i) { Efficient::scan(n, o, i); }, efficientMs},
        {"efficient", "compact", &Efficient::setBlockSize,
         [](int n, int *o, const int *i) { Efficient::compact(n, o, i); }, efficientMs},
        {"efficient_slow", "scan", &EfficientSlow::setBlockSize,
         [](int n, int *o, const int *i) { EfficientSlow::scan(n, o, i); }, efficientSlowMs},
        {"efficient_slow", "compact", &EfficientSlow::setBlockSize,
         [](int n, int *o, const int *i) { EfficientSlow::compact(n, o, i); }, efficientSlowMs},
        {"thrust", "scan", nullptr,
         [](int n, int *o, const int *i) { Thrust::scan(n, o, i); }, thrustMs},
    };
}

// ------------------------------------------------------------- CPU references
// Deliberately not StreamCompaction::CPU:: so that verifying does not clobber
// the CPU module's timer in the middle of a cpu benchmark.

void refScan(int n, int *odata, const int *idata) {
    if (n <= 0) {
        return;
    }
    odata[0] = 0;
    for (int i = 1; i < n; ++i) {
        odata[i] = idata[i - 1] + odata[i - 1];
    }
}

int refCompact(int n, int *odata, const int *idata) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (idata[i] != 0) {
            odata[count++] = idata[i];
        }
    }
    return count;
}

// The run lambdas throw away the element count, so compact is checked over the
// elements the reference says should be there; anything past that is scratch.
bool verifyResult(const char *op, int n, const int *odata, const int *idata, int *ref) {
    if (std::strcmp(op, "scan") == 0) {
        refScan(n, ref, idata);
        return std::equal(ref, ref + n, odata);
    }
    int count = refCompact(n, ref, idata);
    return std::equal(ref, ref + count, odata);
}

// -------------------------------------------------------------- argument bits

std::vector<std::string> splitCsv(const std::string &s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) {
            comma = s.size();
        }
        std::string token = s.substr(start, comma - start);
        if (!token.empty()) {
            out.push_back(token);
        }
        start = comma + 1;
    }
    return out;
}

int parseInt(const std::string &token, const char *what) {
    try {
        size_t used = 0;
        int value = std::stoi(token, &used);
        if (used != token.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string(what) + ": not an integer: '" + token + "'");
    }
}

// "8,12,16", "8..26" (inclusive), or a mix. Result is sorted and deduplicated.
std::vector<int> parseIntList(const std::string &s, const char *what) {
    std::vector<int> out;
    for (const std::string &token : splitCsv(s)) {
        size_t dots = token.find("..");
        if (dots == std::string::npos) {
            out.push_back(parseInt(token, what));
            continue;
        }
        int lo = parseInt(token.substr(0, dots), what);
        int hi = parseInt(token.substr(dots + 2), what);
        if (hi < lo) {
            throw std::invalid_argument(std::string(what) + ": empty range '" + token + "'");
        }
        for (int v = lo; v <= hi; ++v) {
            out.push_back(v);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument(std::string(what) + ": empty list");
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// Small values are far more convenient as exponents than as element counts.
long long asElementCount(int v) {
    return v <= 31 ? (1LL << v) : static_cast<long long>(v);
}

bool selects(const std::vector<std::string> &filter, const char *name) {
    return filter.empty() || std::find(filter.begin(), filter.end(), name) != filter.end();
}

void usage(std::FILE *f) {
    std::fprintf(f,
        "cis5650_stream_compaction_bench - CSV timings for the scan/compact implementations\n"
        "\n"
        "  --impl LIST    all | cpu,cpu_scan,naive,efficient,efficient_slow,thrust (default all)\n"
        "  --op LIST      all | scan,compact                            (default all)\n"
        "  --n LIST       sizes, e.g. 8..26 or 8,12,16; a value <= 31   (default 8..26)\n"
        "                 is a log2 exponent, anything larger is n itself\n"
        "  --block LIST   CUDA block sizes, 1..1024      (default 32,64,128,256,512,1024)\n"
        "  --repeats K    timed runs per configuration                  (default 5)\n"
        "  --warmup K     untimed runs before them                      (default 1)\n"
        "  --maxval V     input values are uniform in [0, V)            (default 4)\n"
        "  --seed S       mt19937 seed for the input                    (default 1234)\n"
        "  --verify       check the first timed run against a CPU reference\n"
        "  --no-header    do not print the CSV header row\n"
        "\n"
        "stdout is CSV: impl,op,n,block_size,run,ms,ok  (block_size 0 = not applicable).\n"
        "Exits 1 if any --verify check failed, 2 on a bad command line.\n");
}

int run(int argc, char *argv[]) {
    std::string implArg = "all";
    std::string opArg = "all";
    std::string nArg = "8..26";
    std::string blockArg = "32,64,128,256,512,1024";
    int repeats = 5;
    int warmup = 1;
    int maxval = 4;
    unsigned int seed = 1234;
    bool verify = false;
    bool header = true;

    auto value = [&](int &i) -> std::string {
        if (i + 1 >= argc) {
            throw std::invalid_argument(std::string(argv[i]) + " needs a value");
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--impl") { implArg = value(i); }
        else if (a == "--op") { opArg = value(i); }
        else if (a == "--n") { nArg = value(i); }
        else if (a == "--block") { blockArg = value(i); }
        else if (a == "--repeats") { repeats = parseInt(value(i), "--repeats"); }
        else if (a == "--warmup") { warmup = parseInt(value(i), "--warmup"); }
        else if (a == "--maxval") { maxval = parseInt(value(i), "--maxval"); }
        else if (a == "--seed") { seed = static_cast<unsigned int>(parseInt(value(i), "--seed")); }
        else if (a == "--verify") { verify = true; }
        else if (a == "--no-header") { header = false; }
        else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
        else { throw std::invalid_argument("unknown argument: " + a); }
    }
    if (repeats < 1) {
        throw std::invalid_argument("--repeats must be at least 1");
    }
    if (warmup < 0) {
        throw std::invalid_argument("--warmup must not be negative");
    }
    if (maxval < 1) {
        throw std::invalid_argument("--maxval must be at least 1");
    }

    std::vector<int> blocks = parseIntList(blockArg, "--block");
    for (int b : blocks) {
        if (b < 1 || b > 1024) {
            throw std::invalid_argument("--block: " + std::to_string(b) + " is outside 1..1024");
        }
    }
    std::vector<int> sizes;
    for (int v : parseIntList(nArg, "--n")) {
        long long n = asElementCount(v);
        if (n < 1 || n > 2147483647LL) {
            throw std::invalid_argument("--n: " + std::to_string(v) + " is out of range");
        }
        sizes.push_back(static_cast<int>(n));
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    std::vector<Benchmark> table = makeTable();
    std::vector<std::string> implFilter =
        implArg == "all" ? std::vector<std::string>() : splitCsv(implArg);
    std::vector<std::string> opFilter =
        opArg == "all" ? std::vector<std::string>() : splitCsv(opArg);
    for (const std::string &name : implFilter) {
        bool known = false;
        for (const Benchmark &b : table) {
            known = known || name == b.impl;
        }
        if (!known) {
            throw std::invalid_argument("--impl: unknown implementation '" + name + "'");
        }
    }
    for (const std::string &name : opFilter) {
        bool known = false;
        for (const Benchmark &b : table) {
            known = known || name == b.op;
        }
        if (!known) {
            throw std::invalid_argument("--op: unknown operation '" + name + "'");
        }
    }
    std::vector<const Benchmark *> selected;
    for (const Benchmark &b : table) {
        if (selects(implFilter, b.impl) && selects(opFilter, b.op)) {
            selected.push_back(&b);
        }
    }
    if (selected.empty()) {
        throw std::invalid_argument("no benchmark matches --impl " + implArg + " --op " + opArg);
    }

    // One allocation at the largest size; every smaller n uses a prefix of it.
    int maxN = sizes.back();
    std::vector<int> idata(maxN);
    std::vector<int> odata(maxN);
    std::vector<int> reference(maxN);
    std::mt19937 gen(seed);
    for (int i = 0; i < maxN; ++i) {
        idata[i] = static_cast<int>(gen() % static_cast<unsigned int>(maxval));
    }

    // Pay for CUDA context creation before anything is timed.
    StreamCompaction::Efficient::scan(std::min(maxN, 1 << 8), odata.data(), idata.data());

    if (header) {
        std::printf("impl,op,n,block_size,run,ms,ok\n");
        std::fflush(stdout);
    }

    int failures = 0;
    for (const Benchmark *b : selected) {
        std::vector<int> groupBlocks = b->setBlock ? blocks : std::vector<int>{0};
        for (int block : groupBlocks) {
            if (b->setBlock) {
                b->setBlock(block);
            }
            std::fprintf(stderr, "%s/%s block=%d: n=%d..%d, %d repeats\n",
                         b->impl, b->op, block, sizes.front(), sizes.back(), repeats);
            for (int n : sizes) {
                for (int w = 0; w < warmup; ++w) {
                    b->run(n, odata.data(), idata.data());
                }
                int ok = 1;
                for (int r = 0; r < repeats; ++r) {
                    b->run(n, odata.data(), idata.data());
                    float ms = b->elapsedMs();
                    if (verify && r == 0) {
                        ok = verifyResult(b->op, n, odata.data(), idata.data(),
                                          reference.data()) ? 1 : 0;
                        if (!ok) {
                            ++failures;
                            std::fprintf(stderr, "  MISMATCH %s/%s n=%d block=%d\n",
                                         b->impl, b->op, n, block);
                        }
                    }
                    std::printf("%s,%s,%d,%d,%d,%.6f,%d\n", b->impl, b->op, n, block, r, ms, ok);
                    std::fflush(stdout);
                }
            }
        }
    }
    if (failures) {
        std::fprintf(stderr, "%d configuration(s) failed verification\n", failures);
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        std::fflush(stdout);
        std::fprintf(stderr, "error: %s\n\n", e.what());
        usage(stderr);
        return 2;
    }
}
