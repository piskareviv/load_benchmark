from os import system
import numpy as np
import re
import argparse
from tqdm import tqdm


modes = ["scalar", "simd_naive", "simd_naive_unaligned", "simd", "simd_unaligned", "simd_x2"]

L, R = (7, 21)
rng = list(map(lambda x: int(round(x)), 2**np.linspace(L, R, (R - L) * 3 + 1)))
use_perf = True


for file in ["A", "B"]:
    ret = system(f"clang++ {file}.cpp -o {file} -std=c++23 -O2 -fno-tree-vectorize -mavx2")
    assert ret == 0
    with open(f"{file}.txt", 'w') as f:
        for tp in modes:
            print(tp, rng, flush=True)
            print(tp, file=f)

            for i in rng:
                if not use_perf:
                    system(f"taskset -c 3 ./A {tp} {i} > tmp.txt")
                    tm = float(open("tmp.txt", 'r').read())
                    print(i, f"{tm:.20f}", file=f, flush=True)
                    print(f"{tm:.20f}", "  ", i, tm, flush=True)

                else:
                    system(f"perf stat -r 2 -d taskset -c 0 ./{file} {tp} {i} > tmp.txt 2> err.txt")

                    s = open("err.txt").read().replace(",", "")

                    c = float(re.findall(r"(\d+)\s+cycles", s)[0])  # cpu cycles
                    if tp == "scalar":
                        c *= 10
                    c = 1e10 / c  # elements per cycle

                    print(i, f"{c:.20f}", file=f, flush=True)
                    print(i, c, flush=True)

            print()

    system("rm tmp.txt")
    system("rm err.txt")
