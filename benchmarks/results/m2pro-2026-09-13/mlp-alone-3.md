## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend amx, Go go1.26.2

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 296.6 µs | 863.1K |
| forward + backward | 1.19 ms | 215.3K |
| forward + backward + Adam step | 1.41 ms | 181.6K |

packed operands: hits 8501, misses 2834, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 31352, misses 353, retained 86 MiB, pinned 0 MiB; GC cycles 34, forced 33
