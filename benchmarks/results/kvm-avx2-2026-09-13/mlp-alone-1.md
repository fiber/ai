## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.65 ms | 155.4K |
| forward + backward | 6.91 ms | 37.0K |
| forward + backward + Adam step | 7.35 ms | 34.8K |

packed operands: hits 1518, misses 502, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 5262, misses 353, retained 28 MiB, pinned 0 MiB; GC cycles 6, forced 5
