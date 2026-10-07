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
    rng = np.random.default_rng()
    V = rng.random(size=(N, M + 1), dtype=np.float32) * 2 - 1
    Vn = V / np.sqrt(np.sum(np.pow(V, 2), axis=0))
    o = np.abs(np.matmul(Vn.T[0:-1], Vn[:, -1]))
    return o


if __name__ == "__main__":
    M = args.M
    batch = args.batch
    nums = [batch for ii in range(M // batch)]  # TODO: smarter chunking
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for N in args.N:
            # TODO: clean up to use map
            results = []
            futs = []
            for ii in range(len(nums)):
                fut = executor.submit(random_products, N, nums[ii], to_print=None)
                futs.append(fut)
            for fut in as_completed(futs):
                results.append(fut.result())
            combined = np.hstack(results)
            #text_hist(combined, bins=round(1.0 / args.binsize), max_width=24)
            mean_result = np.mean(combined)
            print(f"mean: {mean_result}")
            counts, bin_edges = np.histogram(combined, bins=round(1.0 / args.binsize), range=(0, 1))
            output_data = {"N": N, "M": M, "mean": mean_result.astype(float), "hist_counts": counts.tolist(), "hist_bin_edges": bin_edges.tolist()}
            with open(f"out.jsonl", mode="a") as f:
                f.write(json.dumps(output_data)+"\n")
