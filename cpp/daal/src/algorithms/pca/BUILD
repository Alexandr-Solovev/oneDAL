package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    opencl = True,
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:sycl",
        "@onedal//cpp/daal/src/algorithms/covariance:kernel",
        "@onedal//cpp/daal/src/algorithms/svd:kernel",
        "@onedal//cpp/daal/src/algorithms/normalization/zscore:kernel",
    ],
)
