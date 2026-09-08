# BUGS-FIXED

Fixed bugs, newest first. Each points to its spec in `spec/done/`.

- 2026-09-08 B-004 — Masked softmax and attention 11x slower on x86: exp of very negative inputs yields denormal weights (spec/done/B-004-masked-softmax-and-attention-11x-slower-on-x86-e.md)
- 2026-09-05 B-003 — Gate hook depends on the shell working directory and locks the session out after cd (spec/done/B-003-gate-hook-depends-on-the-shell-working-directory.md)
- 2026-09-05 B-002 — Bash hook flags path-like words in heredoc text instead of actual write targets (spec/done/B-002-bash-hook-flags-path-like-words-in-heredoc-text.md)
- 2026-09-05 B-001 — Hook exit code 2 is lost when the gate runs through go run (spec/done/B-001-hook-exit-code-2-is-lost-when-the-gate-runs-thro.md)
