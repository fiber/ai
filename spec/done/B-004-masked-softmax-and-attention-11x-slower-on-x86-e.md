---
id: B-004
title: Masked softmax and attention 11x slower on x86: exp of very negative inputs yields denormal weights
status: done
scope:
  - internal/kernel/
  - tensor/
manual:
  - docs/manual/performance.md
done: 2026-09-08
created: 2026-09-08
---

## Goal

On the Xeon Gold 6130, `tensor.Attention` with a causal mask takes
109 ms for [8×8×512×64] against 9.8 ms without the mask (39 versus
437 GFLOPS); on the M2 Pro both take the same time. Masked softmax in
general is affected. Make masked attention and softmax cost the same as
unmasked on x86.

## Design

Cause: the exp kernels clamp their input at −87, so a masked score
(−1e9 after the row max is subtracted) becomes exp(−87) = 1.6e−38, just
above the smallest normal float. The softmax normalisation then divides
it by the row sum and produces a denormal (about 1e−40). Half the
attention weights under a causal mask are denormals, and the following
weights·V product feeds them to the AVX-512 FMA units, which handle
denormals through microcode assists at roughly a hundred cycles each.
Apple's cores handle denormals at full speed, which is why the M2 never
showed it. Go does not set FTZ/DAZ, and cannot reliably per thread.

Fix in the exp kernels themselves: an input below the clamp threshold
returns exactly 0 instead of exp(−87). That is also the more accurate
answer (the true value is below the normal range), it makes masked
weights exactly zero, and it costs one compare and one blend per vector:

- `generic.go`: `if v < expLo { z[i] = 0 }`.
- AVX2 (`expAVX2`, also used by sigmoid/GELU through the composed
  path): keep the unclamped x, `VCMPPS $1 (LT)` against the lo constant,
  `VANDNPS` the result.
- AVX-512 (if it has its own exp): `VCMPPS` into a k register,
  zero-masking move.
- NEON: `FCMGE x, lo` → mask, `AND` the result; same value semantics
  on all ISAs so results match across back-ends.
- The threshold stays −87.0 (the exponent-add trick's requirement);
  values in (−87, −80] still produce tiny normals whose normalised
  weight may be denormal, but those need a score more than 80 below the
  row maximum and are rare enough not to matter.
- `tensor`: a test that a causal-masked attention row has exact zeros
  where masked, and that a masked and an unmasked call take within 1.5×
  of each other on every back-end (guarded to skip under the race
  detector).

## Acceptance

- `kernel.Exp` returns exactly 0 for inputs below −87 on generic, AVX2,
  AVX-512 and NEON; the existing exp accuracy tests still pass; sigmoid,
  GELU and softmax tests pass.
- Xeon: attention with causal mask within 10 % of the unmasked time
  (target ≥ 400 GFLOPS for [8×8×512×64], baseline PyTorch on the same
  machine to be recorded with the pinned run); M2 numbers unchanged.
- `docs/manual/performance.md` gains a note on denormals and masks.

## Notes

Fixed in the generic, AVX2 (shared by the AVX-512 back-end) and NEON exp
kernels: a compare against the low clamp before clamping, the result
ANDed with the keep mask before the store; one compare and one AND per
vector. Tests: `TestExpFlushesBelowClamp` (exact zeros below −87, no
denormals just above), `TestExpAccuracy` adjusted, and a tensor test
that a causal row's masked softmax weights are exact zeros and position
0 reproduces v[0]. All back-ends pass, including AVX2 under Rosetta.
M2 Pro attention unchanged: 981 GFLOPS unmasked, 898 with the causal
mask. The Xeon confirmation (target: masked within 10 % of unmasked)
comes with the next pinned run; the pinned baseline for both is to be
recorded then.

Xeon confirmation (one socket, pinned, 2026-09-08): attention
[8×8×512×64] 478.5 GFLOPS unmasked, 469.7 with the causal mask (was
39.4). Fixed as specified.

