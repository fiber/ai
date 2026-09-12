# BUGS-FIXED

Fixed bugs, newest first. Each points to its spec in `spec/done/`.

- 2026-09-12 B-011 — bench.py does not parse on the venv's Python: nested quotes in an f-string need 3.12 (spec/done/B-011-bench-py-does-not-parse-on-the-venv-s-python-nes.md)
- 2026-09-12 B-010 — README and getting-started still say the tutorial has seven chapters (spec/done/B-010-readme-and-getting-started-still-say-the-tutoria.md)
- 2026-09-12 B-009 — Conv2D frees the im2col buffer a backward closure still reads (spec/done/B-009-conv2d-frees-the-im2col-buffer-a-backward-closur.md)
- 2026-09-09 B-008 — tutorial chapter 8 prints a pipe table that the chapter quotes inside a code fence, so it renders as text (spec/done/B-008-tutorial-chapter-8-prints-a-pipe-table-that-the.md)
- 2026-09-09 B-007 — tokenizer TestLoadTime asserts a wall-clock bound that a slower or loaded machine misses (spec/done/B-007-tokenizer-testloadtime-asserts-a-wall-clock-boun.md)
- 2026-09-09 B-006 — "smeprobe does not build outside darwin/arm64: missing build constraint breaks go test ./... on Linux" (spec/done/B-006-smeprobe-does-not-build-outside-darwin-arm64-mis.md)
- 2026-09-08 B-005 — Embed pins one encoder output per batch: pool() takes Data(), so Release is refused and memory grows with the number of batches (spec/done/B-005-embed-pins-one-encoder-output-per-batch-pool-tak.md)
- 2026-09-08 B-004 — Masked softmax and attention 11x slower on x86: exp of very negative inputs yields denormal weights (spec/done/B-004-masked-softmax-and-attention-11x-slower-on-x86-e.md)
- 2026-09-05 B-003 — Gate hook depends on the shell working directory and locks the session out after cd (spec/done/B-003-gate-hook-depends-on-the-shell-working-directory.md)
- 2026-09-05 B-002 — Bash hook flags path-like words in heredoc text instead of actual write targets (spec/done/B-002-bash-hook-flags-path-like-words-in-heredoc-text.md)
- 2026-09-05 B-001 — Hook exit code 2 is lost when the gate runs through go run (spec/done/B-001-hook-exit-code-2-is-lost-when-the-gate-runs-thro.md)
