load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_example_suite",
)

dal_module(
    name = "example_util",
    hdrs = glob(["source/example_util/*.hpp"]),
    includes = [ "source" ],
)

_TEST_DEPS = [
    ":example_util",
]

_DATA_DEPS = [
    "@onedal//examples/oneapi:data",
]

dal_example_suite(
    name = "table",
    compile_as = [ "dpc++" ],
    srcs = glob(["source/table/*.cpp"]),
        dpc_deps = [
        "@micromkl//:headers",
        "@micromkl_dpc//:headers",
    ],
)

