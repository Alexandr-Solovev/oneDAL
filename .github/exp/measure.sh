#!/usr/bin/env bash
# Experiment only (not for merging): links dev/release_tests/windows_static_exports against one
# release and reports what the static libraries export into their consumers.
#   measure.sh <label> <release dir> <out dir>
set -euo pipefail
label=$1
release=$(cd "$2/daal/latest" && pwd)
out=$3
mkdir -p "$out"
build=$out/build_$label
libdir=$release/lib/intel64
src=$(cd "$(dirname "$0")/../../dev/release_tests/windows_static_exports" && pwd)

count() { awk -v p="$1" 'tolower($0) ~ p { n++ } END { print n + 0 }'; }

{
    echo "## $label"
    for lib in onedal_core.lib onedal.lib; do
        d=$(dumpbin -nologo -directives "$libdir/$lib" | count '[-/]export:')
        echo "- $lib: $(stat -c %s "$libdir/$lib") bytes, $d /EXPORT directives"
    done
} >> "$out/summary.md"

tbb_dir=()
[ -n "${TBBROOT:-}" ] && [ -d "$TBBROOT/lib/cmake/tbb" ] && tbb_dir=(-DTBB_DIR="$TBBROOT/lib/cmake/tbb")
cmake -S "$src" -B "$build" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DoneDAL_DIR="$libdir/cmake/oneDAL" "${tbb_dir[@]}"
cmake --build "$build"

for b in consumer.exe plugin.dll; do
    p=$build/$b
    dumpbin -nologo -exports "$p" | awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9A-F]+$/ && NF >= 4 { print $4 }' > "$out/${label}_${b}.exports"
    kept=$(count '@daal@@|@dal@oneapi@@' < "$build/${b%.*}.map")
    deps=$(dumpbin -nologo -dependents "$p" | count 'onedal')
    side=$(ls "$build/${b%.*}.exp" "$build/${b%.*}.lib" 2> /dev/null | wc -l)
    echo "- $b: $(stat -c %s "$p") bytes, $(wc -l < "$out/${label}_${b}.exports") exports, $kept oneDAL symbols kept, $deps oneDAL DLL dependents, $side .exp/.lib files" >> "$out/summary.md"
done
export PATH="$build:$release/../../tbb/latest/bin/vc_mt:$PATH"
"$build/consumer.exe" && "$build/plugin_host.exe" && echo "- consumer.exe and plugin_host.exe ran and returned 0" >> "$out/summary.md"
for dll in onedal_core.4.dll onedal.4.dll; do
    dumpbin -nologo -exports "$release/redist/intel64/$dll" > "$out/${label}_$dll.dumpbin"
    echo "- $dll: $(stat -c %s "$release/redist/intel64/$dll") bytes" >> "$out/summary.md"
done
