from os import system
import numpy as np
import re
import argparse
from tqdm import tqdm
 


L, R = (7, 21)
rng = list(map(lambda x: int(round(x)), 2**np.linspace(L, R, (R - L) * 3 + 1)))


ar = ["simd_naive", "simd", "simd_unaligned", "simd_x2"]

for file in ["A", "B"]:
    ret = system(f"g++ {file}.cpp -o {file} -std=c++20 -O2")
    assert ret == 0
    with open(f"{file}.txt", 'w') as f:
        for tp in ar:
            print(tp, rng, flush=True)
            print(tp, file=f)

            for i in rng:
                # system(f"taskset -c 3 ./A {tp} {i} > tmp.txt")
                # tm = float(open("tmp.txt", 'r').read())
                # print(i, f"{tm:.20f}", file=f, flush=True)
                # print(f"{tm:.20f}", "  ", i, tm, flush=True)
            

                system(f"perf stat -r 2 -d taskset -c 0 ./{file} {tp} {i} > tmp.txt 2> err.txt")
            
                s = open("err.txt").read().replace(",", "")
                # print(s)
                c = float(re.findall(r"(\d+)\s+cycles", s)[0]) # cpu cycles
                c = 1e10 / c # elements per cycle

                print(i, f"{c:.20f}", file=f, flush=True)
                print(i, c, flush=True)
                
            print()

    system("rm tmp.txt")
    system("rm err.txt")