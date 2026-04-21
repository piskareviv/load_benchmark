from os import path
import numpy as np
from matplotlib import pyplot as plt



def plot(x, y, label, **kwargs):
    plt.plot(x, y, label=label)

def make_plot(xs, ys, labels, out_file, show=False, y_ticks=1, **kwargs):
    my_dpi = 200
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    plt.yticks(np.arange(0, 100 * y_ticks + 1, y_ticks))

    plt.xticks(np.arange(0, 31, 1))
    plt.grid(linestyle="--")
    plt.axvline(x=13, linestyle="--")
    plt.axvline(x=18, linestyle="--")
    plt.axvline(x=21, linestyle="--")
    plt.axvline(x=0)
    plt.axhline(y=0)

    for x, y, label in zip(xs, ys, labels):
        plot(x, y, label, **kwargs)

    plt.legend()

    if show:
        plt.show()
    # plt.show()

    plt.ylabel("elements per ns")
    plt.xlabel("log_2 n")

    plt.savefig(f"{out_file}.svg")


for s in ["A", "B"]:
    data = open(f"{s}.txt").readlines()

    labels = []
    xs, ys = [], []


    for line in open(f"{s}.txt").readlines():
        if line[0].isnumeric():
            x0, y0 = map(float, line.split())
            xs[-1] += [np.log2(x0)]
            ys[-1] += [y0]
        else:
            labels += [line]
            xs += [[]]
            ys += [[]]

    make_plot(xs, ys, labels, f"plot_{s}")