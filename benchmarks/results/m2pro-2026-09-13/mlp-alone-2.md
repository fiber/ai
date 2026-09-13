## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend amx, Go go1.26.2

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 300.8 µs | 850.9K |
| forward + backward | 1.18 ms | 217.3K |
| forward + backward + Adam step | 1.43 ms | 179.2K |

packed operands: hits 8680, misses 2840, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 31552, misses 353, retained 69 MiB, pinned 0 MiB; GC cycles 34, forced 33
