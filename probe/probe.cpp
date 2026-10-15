// Fits HDBSCAN repeatedly on the same data and reports runs whose medoids differ.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <oneapi/tbb/global_control.h>
#include "oneapi/dal/algo/hdbscan.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace dal = oneapi::dal;
namespace hd = oneapi::dal::hdbscan;

template <typename Method>
std::vector<double> fit(const dal::table& x, hd::store_centers_method sc) {
    auto desc = hd::descriptor<double, Method>(5, 5)
                    .set_store_centers(sc)
                    .set_result_options(hd::result_options::responses | hd::result_options::probabilities |
                                        hd::result_options::cluster_centers | hd::result_options::medoid_centers);
    const auto r = dal::compute(desc, x);
    const auto m = dal::row_accessor<const double>(r.get_medoid_centers()).pull();
    return std::vector<double>(m.get_data(), m.get_data() + m.get_count());
}

int main(int argc, char** argv) {
    const std::string method = argv[2];
    const int runs = atoi(argv[3]);
    const int threads = argc > 4 ? atoi(argv[4]) : 0;
    const auto sc = (argc > 5 && std::string(argv[5]) == "medoid") ? hd::store_centers_method::medoid
                                                                    : hd::store_centers_method::both;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<double> buf((std::istreambuf_iterator<char>(f)), {});
    std::vector<double> data(240 * 3);
    std::ifstream f2(argv[1], std::ios::binary);
    f2.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(double));
    const auto x = dal::homogen_table::wrap(data.data(), 240, 3);
    std::unique_ptr<oneapi::tbb::global_control> gc;
    if (threads > 0)
        gc.reset(new oneapi::tbb::global_control(oneapi::tbb::global_control::max_allowed_parallelism, threads));

    auto run = [&]() {
        if (method == "brute") return fit<hd::method::brute_force>(x, sc);
        if (method == "kd_tree") return fit<hd::method::kd_tree>(x, sc);
        return fit<hd::method::ball_tree>(x, sc);
    };
    const auto ref = run();
    int bad = 0;
    for (int r = 0; r < runs; r++) {
        const auto m = run();
        if (m != ref) {
            if (bad < 5) {
                printf("run %d differs:", r);
                for (size_t k = 0; k < m.size() / 3; k++)
                    if (memcmp(&m[3 * k], &ref[3 * k], 3 * sizeof(double)))
                        printf(" row %zu (%.6f %.6f %.6f) vs ref (%.6f %.6f %.6f)", k, m[3 * k], m[3 * k + 1], m[3 * k + 2],
                               ref[3 * k], ref[3 * k + 1], ref[3 * k + 2]);
                printf("\n");
            }
            bad++;
        }
    }
    printf("method=%s threads=%d centers=%s runs=%d bad=%d ref_row0=(%.6f %.6f %.6f)\n", method.c_str(), threads,
           argc > 5 ? argv[5] : "both", runs, bad, ref[0], ref[1], ref[2]);
    return 0;
}
