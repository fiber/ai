## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend amx, Go go1.26.2

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 291.6 µs | 877.9K |
| forward + backward | 1.20 ms | 213.5K |
| forward + backward + Adam step | 1.45 ms | 176.5K |

packed operands: hits 8641, misses 2800, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 31198, misses 353, retained 184 MiB, pinned 0 MiB; GC cycles 34, forced 33
