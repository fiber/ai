"""PyTorch counterpart of examples/mnist: the same data, shapes, optimiser
and loop, so the two can be timed against each other.

    .venv/bin/python mnist.py [--model mlp|cnn|both] [--epochs 5] [--threads 0]

The digits are read from the github.com/fiber/ai-data module, wherever the
Go tool has put it; --data overrides that.
"""
import argparse, gzip, struct, subprocess, sys, time, platform
import numpy as np
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--model", default="both")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=12)
p.add_argument("--threads", type=int, default=0, help="0 = torch default")
p.add_argument("--data", default="", help="directory holding the four idx.gz files")
p.add_argument("--init", default="torch", choices=("torch", "he"),
               help="torch: kaiming_uniform(a=sqrt(5)) and a uniform bias, the PyTorch default; "
                    "he: N(0, sqrt(2/fan_in)) and a zero bias, what nn.NewLinear/NewConv2D do")
a = p.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)

def data_dir():
    if a.data:
        return a.data.rstrip("/") + "/"
    try:
        out = subprocess.run(["go", "list", "-m", "-f", "{{.Dir}}", "github.com/fiber/ai-data"],
                             capture_output=True, text=True, check=True, cwd="..")
        return out.stdout.strip() + "/mnist/data/"
    except Exception as e:
        sys.exit(f"cannot locate github.com/fiber/ai-data ({e}); pass --data")


DATA = data_dir()


def idx(path):
    with gzip.open(DATA + path, "rb") as f:
        magic, n = struct.unpack(">ii", f.read(8))
        if magic == 0x803:
            rows, cols = struct.unpack(">ii", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
        return np.frombuffer(f.read(), dtype=np.uint8)


def load(images, labels, limit, mean=None, sd=None):
    x = idx(images).astype(np.float32) / 255.0
    y = idx(labels).astype(np.int64)
    if limit:
        x, y = x[:limit], y[:limit]
    if mean is None:
        mean, sd = float(x.mean()), float(x.std())
    x = (x - mean) / sd
    return torch.from_numpy(x), torch.from_numpy(y.copy()), mean, sd


xtr, ytr, mean, sd = load("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", a.limit)
xte, yte, _, _ = load("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 0, mean, sd)
print(f"MNIST: {len(ytr)} training and {len(yte)} test images")
print(f"scaled by mean {mean:.4f} and standard deviation {sd:.4f} of the training pixels")
print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, {platform.processor()}, {a.init} init")


class ToImages(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], 1, 28, 28)


def mlp():
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


def cnn():
    return nn.Sequential(
        ToImages(),
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )


def he_init(model):
    """The initialisation fiber/ai uses, so the two can be compared with
    only the framework differing."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            fan_in = m.weight[0].numel()
            nn.init.normal_(m.weight, 0.0, (2.0 / fan_in) ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


@torch.no_grad()
def evaluate(model, x, y, batch):
    correct = 0
    for s in range(0, len(y), batch):
        correct += (model(x[s:s + batch]).argmax(1) == y[s:s + batch]).sum().item()
    return 100.0 * correct / len(y)


def train(name, model):
    torch.manual_seed(a.seed)
    if a.init == "he":
        he_init(model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossfn = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{name}: {params} parameters, {len(ytr)} images, {a.epochs} epochs of batch {a.batch}")
    g = torch.Generator().manual_seed(a.seed)
    total = 0.0
    for epoch in range(1, a.epochs + 1):
        start = time.perf_counter()
        model.train()
        perm = torch.randperm(len(ytr), generator=g)
        run, steps = 0.0, 0
        for s in range(0, len(ytr), a.batch):
            idxs = perm[s:s + a.batch]
            loss = lossfn(model(xtr[idxs]), ytr[idxs])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            steps += 1
        took = time.perf_counter() - start
        total += took
        model.eval()
        print(f"  epoch {epoch:2d}  loss {run/steps:.4f}  test accuracy {evaluate(model, xte, yte, a.batch):.2f}%  {took:.1f}s")
    model.eval()
    print(f"  {name}: {evaluate(model, xte, yte, a.batch):.2f}% after {total:.1f}s of training")


torch.manual_seed(a.seed)
if a.model in ("mlp", "both"):
    train("MLP  784-256-10", mlp())
if a.model in ("cnn", "both"):
    train("CNN  16-32 filters", cnn())
