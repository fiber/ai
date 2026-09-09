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
parser.add_argument("--only", default="", help="comma-separated sections: gemm,small,elementwise,reductions,mlp,attention,conv,embed")
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


def bench_small():
    """Small products and the tiny autoencoder, the cmd/bench 'small' section."""
    print("### Small products (PyTorch)\n")
    print("| shape | time | GFLOPS |")
    print("|---|---:|---:|")
    with torch.no_grad():
        for n in [32, 64, 96, 128, 160, 192, 256]:
            a, b = torch.randn(n, n), torch.randn(n, n)
            t = time_it(lambda: a @ b)
            print(f"| {n}² | {fmt_dur(t)} | {2 * n**3 / t / 1e9:.1f} |")
        for m, k, n in [(64, 24, 16), (64, 16, 3), (256, 24, 16)]:
            a, b = torch.randn(m, k), torch.randn(k, n)
            t = time_it(lambda: a @ b)
            print(f"| [{m}×{k}]·[{k}×{n}] | {fmt_dur(t)} | {2 * m * n * k / t / 1e9:.1f} |")
    print()
    print("### Tiny autoencoder 24→16→3→16→24, batch 64 (PyTorch)\n")
    print("| phase | time / batch | samples/s |")
    print("|---|---:|---:|")
    batch = 64
    model = torch.nn.Sequential(torch.nn.Linear(24, 16), torch.nn.GELU(), torch.nn.Linear(16, 3),
                                torch.nn.Linear(3, 16), torch.nn.GELU(), torch.nn.Linear(16, 24))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    x = torch.randn(batch, 24)
    loss_fn = torch.nn.MSELoss()

    def inference():
        with torch.no_grad():
            model(x)

    def step():
        with torch.enable_grad():
            loss = loss_fn(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()

    t = time_it(inference)
    print(f"| forward (no_grad) | {fmt_dur(t)} | {fmt_n(batch / t)} |")
    t = time_it(step)
    print(f"| forward + backward + Adam step | {fmt_dur(t)} | {fmt_n(batch / t)} |")
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


def bench_attention():
    """Scaled dot-product attention on [8×8×512×64], the cmd/bench shape."""
    print("### Attention (PyTorch, no_grad)\n")
    print("| shape | time | GFLOPS |")
    print("|---|---:|---:|")
    b, h, n, d = 8, 8, 512, 64
    q, k, v = torch.randn(b, h, n, d), torch.randn(b, h, n, d), torch.randn(b, h, n, d)
    flops = 2.0 * 2 * b * h * n * n * d
    sdpa = torch.nn.functional.scaled_dot_product_attention
    t = time_it(lambda: sdpa(q, k, v))
    print(f"| [{b}×{h}×{n}×{d}] q·kᵀ, softmax, ·v | {fmt_dur(t)} | {flops / t / 1e9:.1f} |")
    t = time_it(lambda: sdpa(q, k, v, is_causal=True))
    print(f"| same with causal mask | {fmt_dur(t)} | {flops / t / 1e9:.1f} |")
    b, n = 1, 2048
    q, k, v = torch.randn(b, h, n, d), torch.randn(b, h, n, d), torch.randn(b, h, n, d)
    flops = 2.0 * 2 * b * h * n * n * d
    t = time_it(lambda: sdpa(q, k, v))
    print(f"| [{b}×{h}×{n}×{d}] long sequence | {fmt_dur(t)} | {flops / t / 1e9:.1f} |")
    print()


def bench_conv():
    """3×3 convolution on [32×64×56×56] with 64 filters, padding 1."""
    print("### Convolution (PyTorch, no_grad)\n")
    print("| shape | time | GFLOPS |")
    print("|---|---:|---:|")
    n, c, h, w, o, k = 32, 64, 56, 56, 64, 3
    x = torch.randn(n, c, h, w)
    wt = torch.randn(o, c, k, k)
    bias = torch.randn(o)
    flops = 2.0 * n * o * h * w * c * k * k
    t = time_it(lambda: torch.nn.functional.conv2d(x, wt, bias, stride=1, padding=1))
    print(f"| [{n}×{c}×{h}×{w}] · {o} filters {k}×{k}, pad 1 | {fmt_dur(t)} | {flops / t / 1e9:.1f} |")
    print()


def bench_embed():
    """EmbeddingGemma via sentence-transformers, fp32, 32 sentences of ~64 tokens."""
    import os
    print("### EmbeddingGemma (sentence-transformers, fp32, 32 sentences ≈ 64 tokens, one batch)\n")
    root = os.environ.get("FIBERAI_MODELS")
    if not root:
        print("skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m\n")
        return
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("skipped: sentence-transformers is not installed\n")
        return
    model = SentenceTransformer(os.path.join(root, "embeddinggemma-300m"), device="cpu").float()
    sentence = ("the quick brown fox jumps over the lazy dog " * 7).strip()
    texts = [sentence] * 32
    ntok = len(model.tokenizer(sentence)["input_ids"])
    enc = lambda: model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    t = time_it(enc)
    print(f"threads {torch.get_num_threads()}\n")
    print("| shape | time / batch | sentences/s |")
    print("|---|---:|---:|")
    print(f"| 32 × {ntok} tokens, dim {(model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension())} | {fmt_dur(t)} | {32 / t:.0f} |")
    print()


SECTIONS = {"gemm": bench_gemm, "elementwise": bench_elementwise, "reductions": bench_reductions,
            "mlp": bench_mlp, "attention": bench_attention, "conv": bench_conv, "embed": bench_embed, "small": bench_small}
order = [s.strip() for s in args.only.split(",") if s.strip()] or ["gemm", "small", "elementwise", "reductions", "attention", "conv", "mlp", "embed"]
for name in order:
    if name not in SECTIONS:
        sys.exit(f"unknown section {name!r}")
    SECTIONS[name]()
