
from os import path
import numpy as np
from matplotlib import pyplot as plt

from dataclasses import dataclass
import argparse
from os import system
import numpy as np
import re
import argparse
from tqdm import tqdm





TOTAL_SIZE = 10**10
L, R = (7, 21)


def run_benchmark(compile_line, file, modes):

    rng = list(map(lambda x: int(round(x)), 2**np.linspace(L, R, (R - L) * 3 + 1)))

    ret = system(compile_line)
    assert ret == 0

    results = []

    for mode in modes:
        print(f"mode: {mode},  total {len(rng)} sizes, from {min(rng)} to {max(rng)}", flush=True)

        res = []
        for n in rng:
            # system(f"taskset -c 3 ./A {tp} {i} > tmp.txt")
            # tm = float(open("tmp.txt", 'r').read())
            # print(i, f"{tm:.20f}", file=f, flush=True)
            # print(f"{tm:.20f}", "  ", i, tm, flush=True)



            iters = TOTAL_SIZE // n

            if mode == "scalar":
                iters //= 10

            ret = system(f"perf stat -r 2 -d taskset -c 0 ./{file} {mode} {n} {iters} > tmp.txt 2> err.txt")
            assert ret == 0

            s = open("err.txt").read().replace(",", "")
            cycles = float(re.findall(r"(\d+)\s+cycles", s)[0])  # cpu cycles
            performance = n * iters / cycles  # elements per cycle


            print(f"{str(n).ljust(10)} {performance:.3f}")

            res += [(n, performance)]


        results += [res]

        print()

    system("rm tmp.txt")
    system("rm err.txt")

    return results




def plot_benchmark_results(results, compile_line, proc_model, file, output_file, modes, y_ticks=2):


    my_dpi = 200
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

    ax.set_title(f'compile: {compile_line}\nproc: {proc}')

    ax.set_xlabel("log_2 n")
    ax.set_ylabel("elements per ns")


    ax.set_xticks(np.arange(0, 31, 1))
    ax.set_yticks(np.arange(0, 100 * y_ticks + 1, y_ticks))

    ax.set_xlim((L-0.5, R+0.5))
    ax.set_ylim((0-0.5, 16+0.5))
    
    ax.grid(linestyle="--")
    ax.axvline(x=13, linestyle="--")
    ax.axvline(x=18, linestyle="--")
    ax.axvline(x=21, linestyle="--")
    ax.axvline(x=0)
    ax.axhline(y=0)

    for mode, res in zip(modes, results):
        x, y = zip(*res)
        ax.plot(np.log2(x), y, label=mode)

    ax.legend()



    fig.savefig(f"{output_file}.svg")


def run_and_plot(compile_line, proc_model, file, output_file, modes, **kwargs):
    results = run_benchmark(compile_line=compile_line, file=file, modes=modes)
    plot_benchmark_results(results=results, compile_line=compile_line, proc_model=proc_model, file=file, output_file=output_file, modes=modes, **kwargs)



def compile_gcc(file):
    return f"g++ {file}.cpp -o {file} -std=c++23 -O2 -mavx2"

def compile_clang(file):
    return f"clang++ {file}.cpp -o {file} -std=c++23 -O2 -mavx2"




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("proc_model")
    args = parser.parse_args()
    proc = args.proc_model



    files = ["A", "B"]
    modes = ["scalar", "simd_naive", "simd_naive_unaligned", "simd", "simd_unaligned", "simd_x2"]
    

    for file in ["A", "B"]:
        run_and_plot(compile_gcc(file), proc_model=proc, file=file, output_file=f"{file}_gcc_{proc}".replace(" ", "-"), modes=modes)
        run_and_plot(compile_clang(file), proc_model=proc, file=file, output_file=f"{file}_clang_{proc}".replace(" ", "-"), modes=modes)



