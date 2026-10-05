import multiprocessing
from argparse import ArgumentParser
import numpy as np
from time import sleep
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = ArgumentParser()
parser.add_argument('--workers', nargs='?', help='workers help', default=4, type=int)
parser.add_argument('--batch', nargs='?', default=10_000, type=int)
parser.add_argument('--N', nargs='?', default=2048, type=int)
parser.add_argument('--M', nargs='?', default=10_000_000, type=int)
args = parser.parse_args()
print(args)

def text_hist(data, bins=10, max_width=40):
    counts, bin_edges = np.histogram(data, bins=bins)
    max_count = np.log(max(counts)) if max(counts) > 0 else 1
    print("--- NumPy Terminal Histogram ---")
    for i in range(len(counts)):
        bar_length = int(np.log(counts[i]) / max_count * max_width)
        bar = "█" * bar_length
        label = f"[{bin_edges[i]:1.3f} : {bin_edges[i+1]:1.3f}]"
        print(f"{label} | {bar} {counts[i]}")


def random_products(M, to_print=None):
    if to_print is not None:
        print(to_print)
    rng = np.random.default_rng()
    V = rng.random(size=(N,M+1), dtype=np.float32) * 2 - 1
    Vn = V/np.sqrt(np.sum(np.pow(V,2), axis=0))
    o = np.abs(np.matmul(Vn.T[0:-1], Vn[:,-1]))
    return o
#
# if __name__ == "__main__":
#     N = 2048
#     M = 1_000_000
#     batch = 10_000
#     nums = [batch for ii in range(M//batch)] # TODO smarter chunking
#     with ProcessPoolExecutor(max_workers=8) as executor:
#         results = list(executor.map(random_products, nums))
#     combined = np.hstack(results)
#     text_hist(combined, max_width=24)


if __name__ == "__main__":
    N = args.N
    M = args.M
    batch = args.batch
    nums = [batch for ii in range(M//batch)] # TODO smarter chunking
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as executor:
        #results = list(executor.map(random_products, nums))
        results = []
        futs = []
        for ii in range(len(nums)):
            fut = executor.submit(random_products, nums[ii], to_print=None)
            futs.append(fut)
        print("submitted!")
        for fut in as_completed(futs):
            results.append(fut.result())
    combined = np.hstack(results)
    #text_hist(combined, max_width=24)
