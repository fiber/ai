package kernel

import (
	"math"
	"testing"
)

// TestExpFlushesBelowClamp: inputs below the low clamp return exactly 0 on
// the active back-end and on the generic kernel, so masked softmax weights
// are exact zeros and never denormals (B-004).
func TestExpFlushesBelowClamp(t *testing.T) {
	in := make([]float32, 64)
	for i := range in {
		switch {
		case i%4 == 0:
			in[i] = -1e9 // a masked score
		case i%4 == 1:
			in[i] = -87.0001
		case i%4 == 2:
			in[i] = -86.9 // just above: a tiny normal
		default:
			in[i] = float32(i) * 0.25
		}
	}
	for name, fn := range map[string]func(x, z []float32){"active": Exp, "generic": genericExp} {
		out := make([]float32, len(in))
		fn(in, out)
		for i, x := range in {
			bits := math.Float32bits(out[i])
			switch {
			case x < expLo:
				if bits != 0 {
					t.Errorf("%s: exp(%v) = %g (bits %#x), want exactly 0", name, x, out[i], bits)
				}
			case bits&0x7f800000 == 0 && bits != 0:
				t.Errorf("%s: exp(%v) = %g is denormal", name, x, out[i])
			default:
				want := float32(math.Exp(float64(x)))
				if math.Abs(float64(out[i]-want)) > 2e-6*float64(want) {
					t.Errorf("%s: exp(%v) = %g, want %g", name, x, out[i], want)
				}
			}
		}
	}
}
