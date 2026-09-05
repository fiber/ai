"""Python counterpart of cmd/bench: the same workloads with NumPy and PyTorch.

    .venv/bin/python bench.py [--quick]
"""
import argparse
import platform
import sys
import time

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
parser.add_argument("-d", type=float, default=0.7, help="seconds per case")
args = parser.parse_args()
DURATION = args.d

torch.set_grad_enabled(False)


def time_it(fn):
    fn()
    iters = 0
    start = time.perf_counter()
    while time.perf_counter() - start < DURATION:
        fn()
        iters += 1
    return (time.perf_counter() - start) / iters


def fmt_dur(sec):
    if sec < 1e-3:
        return f"{sec*1e6:.1f} µs"
    if sec < 1:
        return f"{sec*1e3:.2f} ms"
    return f"{sec:.2f} s"


def fmt_n(n):
    if n >= 1 << 20 and n % (1 << 20) == 0:
        return f"{n >> 20}M"
    if n >= 1 << 10 and n % (1 << 10) == 0:
        return f"{n >> 10}K"
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if n >= 1000:
        return f"{n/1e3:.1f}K"
    return str(n)


print(f"## Python — {platform.system().lower()}/{platform.machine()}, "
      f"NumPy {np.__version__}, PyTorch {torch.__version__} "
      f"({torch.get_num_threads()} threads), Python {sys.version.split()[0]}\n")
blas = "unknown"
for line in torch.__config__.show().splitlines():
    if "BLAS" in line or "blas" in line.lower():
        blas = line.strip()
        break
print(f"torch BLAS backend: {blas}\n")


def bench_gemm():
    sizes = [128, 256, 512, 1024] if args.quick else [128, 256, 512, 1024, 2048]
    print("### Matrix multiply (float32, GFLOPS)\n")
    print("| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |")
    print("|---:|---:|---:|---:|")
    all_threads = torch.get_num_threads()
    for n in sizes:
        a = np.random.randn(n, n).astype(np.float32)
        b = np.random.randn(n, n).astype(np.float32)
        ta, tb = torch.from_numpy(a), torch.from_numpy(b)
        flops = 2 * n ** 3
        t_np = time_it(lambda: a @ b)
        torch.set_num_threads(1)
        t_t1 = time_it(lambda: ta @ tb)
        torch.set_num_threads(all_threads)
        t_tn = time_it(lambda: ta @ tb)
        print(f"| {n} | {flops/t_np/1e9:.1f} | {flops/t_t1/1e9:.1f} | {flops/t_tn/1e9:.1f} |")
    print()
    print("| shape | NumPy GFLOPS | PyTorch GFLOPS |")
    print("|---|---:|---:|")
    for m, k, n in [(1, 4096, 4096), (8, 4096, 4096), (64, 1024, 1024), (256, 768, 3072)]:
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        ta, tb = torch.from_numpy(a), torch.from_numpy(b)
        flops = 2 * m * n * k
        t_np = time_it(lambda: a @ b)
        t_t = time_it(lambda: ta @ tb)
        print(f"| [{m}×{k}]·[{k}×{n}] | {flops/t_np/1e9:.1f} | {flops/t_t/1e9:.1f} |")
    a = np.random.randn(1024, 1024).astype(np.float32)
    b = np.random.randn(1024, 1024).astype(np.float32)
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    t_np = time_it(lambda: a @ b.T)
    t_t = time_it(lambda: ta @ tb.T)
    print(f"| [1024×1024]·[1024×1024]ᵀ (view) | {2*1024**3/t_np/1e9:.1f} | {2*1024**3/t_t/1e9:.1f} |")
    print()


def bench_elementwise():
    print("### Element-wise\n")
    print("| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |")
    print("|---|---:|---:|---:|---:|---:|")
    for n in [1 << 16, 1 << 20, 1 << 24]:
        x = np.random.randn(n).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        row = np.random.randn(1024).astype(np.float32)
        x2 = x.reshape(-1, 1024)
        tx, ty, trow, tx2 = map(torch.from_numpy, (x, y, row, x2))
        cases = [
            ("x + y", 12 * n, lambda: x + y, lambda: tx + ty),
            ("x * y", 12 * n, lambda: x * y, lambda: tx * ty),
            ("x + row (broadcast)", 8 * n, lambda: x2 + row, lambda: tx2 + trow),
            ("x * 2.5", 8 * n, lambda: x * 2.5, lambda: tx * 2.5),
            ("exp(x)", 8 * n, lambda: np.exp(x), lambda: torch.exp(tx)),
            ("tanh(x)", 8 * n, lambda: np.tanh(x), lambda: torch.tanh(tx)),
            ("relu(x)", 8 * n, lambda: np.maximum(x, 0), lambda: torch.relu(tx)),
        ]
        for name, nbytes, f_np, f_t in cases:
            t_np, t_t = time_it(f_np), time_it(f_t)
            print(f"| {name} | {fmt_n(n)} | {fmt_dur(t_np)} | {nbytes/t_np/1e9:.1f} | "
                  f"{fmt_dur(t_t)} | {nbytes/t_t/1e9:.1f} |")
    print()


def bench_reductions():
    print("### Reductions\n")
    print("| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |")
    print("|---|---|---:|---:|---:|---:|")
    x = np.random.randn(4096, 4096).astype(np.float32)
    tx = torch.from_numpy(x)
    n = x.size
    g, b = torch.ones(4096), torch.zeros(4096)
    cases = [
        ("sum()", 4 * n, lambda: x.sum(), lambda: tx.sum()),
        ("sum(dim=0)", 4 * n, lambda: x.sum(0), lambda: tx.sum(0)),
        ("sum(dim=1)", 4 * n, lambda: x.sum(1), lambda: tx.sum(1)),
        ("max(dim=1)", 4 * n, lambda: x.max(1), lambda: tx.max(1)),
        ("softmax(dim=1)", 8 * n, None, lambda: torch.softmax(tx, 1)),
        ("layernorm", 8 * n, None, lambda: torch.nn.functional.layer_norm(tx, (4096,), g, b, 1e-5)),
        ("transpose+contiguous", 8 * n, lambda: np.ascontiguousarray(x.T), lambda: tx.T.contiguous()),
    ]
    for name, nbytes, f_np, f_t in cases:
        t_t = time_it(f_t)
        if f_np is None:
            np_cols = "– | –"
        else:
            t_np = time_it(f_np)
            np_cols = f"{fmt_dur(t_np)} | {nbytes/t_np/1e9:.1f}"
        print(f"| {name} | [4096×4096] | {np_cols} | {fmt_dur(t_t)} | {nbytes/t_t/1e9:.1f} |")
    print()


def bench_mlp():
    print("### MLP 784→512→512→10, batch 256 (PyTorch)\n")
    print("| phase | time / batch | samples/s |")
    print("|---|---:|---:|")
    batch = 256
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch, 784)
    targets = torch.arange(batch) % 10
    loss_fn = torch.nn.CrossEntropyLoss()

    def inference():
        with torch.no_grad():
            model(x)

    def fwd_bwd():
        with torch.enable_grad():
            loss = loss_fn(model(x), targets)
            opt.zero_grad()
            loss.backward()

    def step():
        with torch.enable_grad():
            loss = loss_fn(model(x), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()

    for name, fn in [("forward (no_grad)", inference), ("forward + backward", fwd_bwd),
                     ("forward + backward + Adam step", step)]:
        t = time_it(fn)
        print(f"| {name} | {fmt_dur(t)} | {fmt_n(int(batch/t))} |")
    print()


bench_gemm()
bench_elementwise()
bench_reductions()
bench_mlp()
