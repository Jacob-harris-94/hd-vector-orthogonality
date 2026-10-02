import numpy as np

def text_hist(data, bins=10, max_width=40):
    # Calculate intervals and frequencies using NumPy
    counts, bin_edges = np.histogram(data, bins=bins)
    max_count = np.log(max(counts)) if max(counts) > 0 else 1
    
    print("--- NumPy Terminal Histogram ---")
    for i in range(len(counts)):
        # Calculate visual bar scaling to prevent terminal overflow
        bar_length = int(np.log(counts[i]) / max_count * max_width)
        bar = "█" * bar_length
        
        # Format strings for aligned printing
        label = f"[{bin_edges[i]:1.3f} : {bin_edges[i+1]:1.3f}]"
        print(f"{label} | {bar} {counts[i]}")

# # Example Usage:
# np.random.seed(42)
# normal_data = np.random.normal(loc=50, scale=10, size=100)
# numpy_text_hist(normal_data, bins=8)
#

rng = np.random.default_rng()

N = 2048
M = 1_000_000
V = rng.random(size=(N,M), dtype=np.float32) * 2 - 1
Vn = V/np.sqrt(np.sum(np.pow(V,2), axis=0))

#f = np.abs(np.matmul(Vn.T, Vn))
o = np.abs(np.matmul(Vn.T[0:-1], Vn[:,-1]))
#f = f[~np.eye(f.shape[0], dtype=bool)] # lose main diagonal

print(o.shape, o.mean())
#print(f.shape, f.mean())

text_hist(o, max_width=24)
#text_hist(f, max_width=24)
