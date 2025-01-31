#!/bin/bash

rm -rf build
mkdir build && cd build
mkdir logs
cmake ..
make -j

for executable in ./gemm_*
do
    if [[ -x "$executable" ]]; then
        echo "*** Executing $executable ***"
        "$executable" | tee "logs/${executable}.log"
    else
        echo "[Skipping] $executable (not executable)"
    fi
done

cd ..
python compare_performance.py
