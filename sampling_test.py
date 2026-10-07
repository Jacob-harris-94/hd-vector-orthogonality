import json
from argparse import ArgumentParser
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = ArgumentParser()
parser.add_argument("--workers", nargs="?", help="workers help", default=4, type=int)
parser.add_argument("--batch", nargs="?", default=10_000, type=int)
parser.add_argument("--binsize", nargs="?", default=0.05, type=float)
parser.add_argument("--N", nargs="+", default=[], type=int)
parser.add_argument("--M", nargs="?", default=10_000_000, type=int)
args = parser.parse_args()
print(args)


def text_hist(data, bins=10, max_width=40):
    counts, bin_edges = np.histogram(data, bins=bins, range=(0, 1))
    max_count = np.log(max(counts)) if max(counts) > 0 else 1
    print("-" * max_width)
    for i in range(len(counts)):
        bar_length = (
            int(np.log(counts[i]) / max_count * max_width) if counts[i] >= 1 else 0
        )
        bar = "█" * bar_length
        label = f"[{bin_edges[i]:1.3f} : {bin_edges[i+1]:1.3f}]"
        print(f"{label} | {bar} {counts[i]}")


def random_products(N, M, to_print=None):
    if to_print is not None:
        print(to_print)
    N = 2
    rng = np.random.default_rng()
    V = rng.random(size=(N, M + 1), dtype=np.float32) * 2 - 1
    Vn = V / np.sqrt(np.sum(np.pow(V, 2), axis=0))
    angles = np.atan(Vn[1,:]/Vn[0,:])
    # ignore above
    #return angles
    A = rng.random(size=(1, M), dtype=np.float32) * 2 * np.pi # confirmed good with occular guestimation
    Vn = np.vstack([np.cos(A), np.sin(A)]) # evenly distributed by construction
    #angles = np.atan(Vn[1,:]/Vn[0,:]) confirmed even
    o = np.abs(np.matmul(Vn.T[0:-1], Vn[:, -1]))
    return o

if __name__ == "__main__":
    M = args.M
    batch = args.batch
    nums = [batch for ii in range(M // batch)]  # TODO: smarter chunking
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for N in args.N:
            combined = random_products(N, M)
            text_hist(combined, bins=20, max_width=24)
