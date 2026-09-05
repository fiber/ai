# fiber/ai manual

The user manual for the fiber/ai Go AI framework. Kept current by the
development process: every spec names the pages it changes and the gate
refuses to complete a spec whose pages were not updated (see
[process.md](process.md)).

1. [Getting started](getting-started.md) — install, first program, running the examples
2. [Tensors](tensors.md) — creation, shapes, views, broadcasting, indexing, printing, errors
3. [Autograd](autograd.md) — gradients, Backward, NoGrad, in-place rules, gradient checking
4. [Neural networks and optimisers](nn-and-optim.md) — modules, losses, training loops, SGD/Adam
5. [Performance](performance.md) — back-ends, threads, environment variables, what is fast and what is not
6. [Internals](internals.md) — kernels, GEMM blocking, adding an operation or an architecture
7. [Application packages](applications.md) — `logtemplate` (syslog templates at millions of lines a second), `cluster` (k-means and nearest centre over embeddings)
8. [Development process](process.md) — TODO/BUGS/DONE, specs, the gate tool
