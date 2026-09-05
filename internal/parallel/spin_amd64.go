package parallel

import "time"

// On x86 servers a longer spin pays: the Xeon Gold 6130 training step
// went 49K → 53K samples/s from 300 µs to 3 ms, with no cost while busy.
const defaultSpin = 2 * time.Millisecond
