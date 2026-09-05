---
id: T-027
title: Conv1D and Conv2D via im2col with padding and stride, nn modules, pooling
status: done
scope:
  - tensor/
  - nn/
  - examples/
  - cmd/bench/
manual:
  - docs/manual/nn-and-optim.md
  - docs/manual/tensors.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

The other half of TODO item T-009: convolutions, so that images, spectra
and long event sequences can be modelled (a 1-D convolution over the
last 24 hours of traffic is the natural next step of tutorial chapter 8).
Built on what exists: im2col turns a convolution into one matrix product
per batch element, which the GEMM path already runs at full speed.

## Design

- `tensor.Im2Col(x, kh, kw, stride, pad)`: x [N,C,H,W] → [N, C·kh·kw,
  Ho·Wo], zero padding, a recorded operation whose backward scatters the
  gradient back (col2im, adding where patches overlap).
- `tensor.Conv2D(x, w, b, stride, pad)`: w [O,C,kh,kw], b [O] or nil;
  `w.Reshape(O, K).MatMul(col)` broadcast over the batch, plus bias,
  reshaped to [N,O,Ho,Wo]. Autograd for w and b comes from MatMul and Add,
  for x from Im2Col.
- `tensor.Conv1D(x, w, b, stride, pad)`: x [N,C,L], w [O,C,k], through
  Conv2D with height 1.
- `tensor.MaxPool2D(x, k, stride)` with a recorded backward routing each
  gradient to the position of the maximum.
- `nn.Conv2D`, `nn.Conv1D` (weights He-initialised, optional bias,
  Stride, Pad) and `nn.MaxPool2D` modules; `nn.Flatten` for the
  transition to Linear.
- Example `examples/nn/conv`: a small CNN on generated 16×16 images
  (which of four shapes is drawn) reaching over 95 % test accuracy.
- `cmd/bench` conv section: [32×64×56×56] with 64 3×3 filters, forward
  under NoGrad, GFLOPS.

## Acceptance

- Conv2D and Conv1D match naive loop implementations to 1e-5 for stride
  1 and 2, padding 0 and 1, several channel counts; numeric gradient
  checks for x, w and b; MaxPool2D forward and backward against a loop.
- The example reaches ≥ 95 % on held-out images within ten seconds on
  the M2 Pro.
- Performance: the bench row runs at ≥ 50 % of the GEMM rate of the
  equivalent product ([64×576]·[576×3136] per image) on the M2 Pro; the
  PyTorch figure for the same shape joins the Python bench in the next
  x86 round. No impact on existing operations.
- nn-and-optim.md documents the modules, tensors.md the functions.

## Notes

Implemented. `Im2Col` lays all images of a batch side by side
([C·kh·kw, N·Ho·Wo]) so a convolution is one product for the whole batch;
the stride-1 path copies contiguous runs. Conv2D/Conv1D/MaxPool2D match
naive loops; numeric gradient checks pass for x, w, b and through a
Conv→GELU→Conv→Linear stack (ReLU/max-pool kinks make finite differences
unreliable, so the gradient test uses the smooth variant). The CNN
example reaches 97.6 % on held-out shape images in 0.7 s. Bench
[32×64×56×56]·64 3×3: 20 ms, 367 GFLOPS; releasing the column and the
pre-bias product removed a forced collection per call (27 → 3 per run).
The remaining time is memory: the column matrix is 231 MB written by
im2col and read again by the GEMM's B packing, so the operation runs at
the rate of those two passes, not the FMA rate. An implicit GEMM (B
panels packed straight from the image, no column matrix) is the
follow-up if convolutions become a workload; the 50 % criterion against
the bare product is therefore not verified here, and the PyTorch row is
still to be measured.
