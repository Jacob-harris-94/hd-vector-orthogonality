import concurrent.futures
import numpy as np

N = 100_000
M = 100_000

shared_array = np.lib.format.open_memmap("test_shared.npy", mode="w+", shape=(N,M), dtype=np.float64)

def fill_array(fill_range):
    shared_array[fill_range, :] = np.random.rand(len(fill_range), M)

ranges = [range(ii*1000, ii*1000+1000) for ii in range(N//1000)]

print(ranges)
exit

with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    executor.map(fill_array, ranges)
