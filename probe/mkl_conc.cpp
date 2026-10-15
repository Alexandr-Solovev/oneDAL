// Concurrent sequential dgemm calls, as in the HDBSCAN medoid kernel, checked against a naive product.
#include <mkl.h>
#include <oneapi/tbb.h>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
int main(int argc, char** argv) {
    const int reps = atoi(argv[1]);
    const MKL_INT sizes[4] = { 41, 63, 60, 62 }, k = 3;
    std::vector<std::vector<double>> a(4);
    std::mt19937 g(0);
    std::normal_distribution<double> nd;
    for (int t = 0; t < 4; t++) {
        a[t].resize(sizes[t] * k);
        for (auto& v : a[t]) v = nd(g);
    }
    std::atomic<long> bad{ 0 };
    for (int r = 0; r < reps; r++) {
        tbb::parallel_for(0, 4, 1, [&](int t) {
            const MKL_INT m = sizes[t];
            std::vector<double> out(m * m);
            const double one = 1, zero = 0;
            int old = mkl_set_num_threads_local(1);
            dgemm("t", "n", &m, &m, &k, &one, a[t].data(), &k, a[t].data(), &k, &zero, out.data(), &m);
            mkl_set_num_threads_local(old);
            for (MKL_INT i = 0; i < m; i++)
                for (MKL_INT j = 0; j < m; j++) {
                    double s = 0;
                    for (MKL_INT d = 0; d < k; d++) s += a[t][i * k + d] * a[t][j * k + d];
                    if (std::fabs(s - out[i * m + j]) > 1e-12 * (1 + std::fabs(s))) { bad++; return; }
                }
        }, tbb::simple_partitioner());
    }
    printf("mkl concurrent dgemm: reps=%d bad=%ld\n", reps, bad.load());
}
