//go:build !amd64

package parallel

import "time"

// Apple's performance/efficiency mix punishes long spins (a 1 ms spin
// cost the M2 Pro 20 % on the training step): keep it short.
const defaultSpin = 300 * time.Microsecond
