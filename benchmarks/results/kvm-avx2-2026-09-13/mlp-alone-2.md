## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.84 ms | 139.0K |
| forward + backward | 7.08 ms | 36.1K |
| forward + backward + Adam step | 8.28 ms | 30.9K |

packed operands: hits 1356, misses 478, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 4902, misses 353, retained 97 MiB, pinned 0 MiB; GC cycles 6, forced 5
