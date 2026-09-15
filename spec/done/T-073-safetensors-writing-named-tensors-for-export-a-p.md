---
id: T-073
title: safetensors: writing named tensors, for export a Python reader can open
status: done
scope:
  - safetensors/
  - docs/manual/nn-and-optim.md
  - docs/manual/models.md
manual:
  - docs/manual/nn-and-optim.md
  - docs/manual/models.md
done: 2026-09-15
created: 2026-09-15
---

---

## Goal

The library can read safetensors and cannot write it. The only way out
of a trained model is `nn.SaveParams`, which writes our own `FAIP`
format: positional, unnamed, no metadata, readable only by a Go program
that builds the identical architecture first. Anyone who wants to look
at a trained model from Python, keep weights next to a model card, or
hand a checkpoint to a colleague on a different stack has nothing.

The gap showed up in an agent's forecasting run, which had to write the
format by hand. Writing it is fifty lines — the format is a header
length, a JSON header and the raw data — and the reader is already
here, which makes the absence harder to justify than the work.

## Design

`safetensors.Save(path, tensors, meta)` and `safetensors.Write(w,
tensors, meta)`, taking `map[string]*tensor.Tensor` and an optional
`map[string]string` of metadata that becomes the `__metadata__` entry.
Everything is written as `F32`, which is what a `tensor.Tensor` holds;
the reader's other dtypes stay read-only and the documentation says so.

- **Deterministic output.** Tensors are written in sorted name order
  and `encoding/json` sorts the header's keys, so saving the same model
  twice gives identical bytes and a checksum means something.
- **Alignment.** The header is padded with spaces so the data section
  starts at a multiple of 8 bytes, which is what the reference
  implementation does and what memory-mapping readers expect.
- **Data offsets** are relative to the start of the data section, as
  the format defines and as our reader already assumes.
- Non-contiguous tensors are materialised in row-major order through
  `CopyTo`, so a transposed view saves as what it looks like.
- Names must be non-empty and must not be `__metadata__`.

`Write` builds the header, then streams the tensors in order; nothing
holds a second copy of the model.

## Acceptance

- Round trip: save a set of tensors of different ranks, open the file
  with this package, get the same names, shapes and values bit for bit.
- The file a Python reader sees is valid: a stdlib-only parser
  (`struct` + `json`, no numpy) reads the header, finds the data
  section 8-byte aligned, and recovers every tensor's bytes at the
  offsets the header names.
- Saving the same tensors twice produces byte-identical files.
- A tensor saved from a transposed view reads back as the transposed
  values, not the original layout.
- Errors: an empty name, the reserved `__metadata__` name and a nil
  tensor are refused.
- No performance impact: a new write path, no change to reading or to
  any kernel.

## Notes

Verified against an outside reader as well as our own: a stdlib-only
Python parser (`struct` + `json`, no numpy, no safetensors package)
reads a file written here — header 392 bytes, data section starting at
400 and so 8-byte aligned, metadata intact, all five tensors recovered
with their shapes and values.

Two deliberate limits, both documented. Writing is F32 only, because
that is what a `tensor.Tensor` holds; reading keeps F16, BF16, F64 and
the integer types. And there is no convenience that exports a
`nn.Parameterised` directly, because `Params()` has no names and
inventing `param.0`, `param.1` would produce a file that is named but
not meaningful. The caller names the tensors.
