## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.41 ms | 181.6K |
| forward + backward | 7.19 ms | 35.6K |
| forward + backward + Adam step | 8.13 ms | 31.5K |

packed operands: hits 1618, misses 482, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 5194, misses 353, retained 86 MiB, pinned 0 MiB; GC cycles 6, forced 5
