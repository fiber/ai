package gemma

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/cmplx"
)

// The audio frontend of EmbeddingGemma 2 (Gemma 4's feature extractor):
// 16 kHz samples to log-mel frames, 128 bins every 10 ms.

const (
	audioRate       = 16000
	audioMaxSamples = 30 * audioRate // the extractor truncates longer audio
	melFrame        = 320            // 20 ms
	melHop          = 160            // 10 ms
	melFFT          = 512
	melBins         = 128
	melMaxHz        = 8000
	melFloor        = 1e-3
	audioPadTo      = 128 // the extractor pads to a multiple of this
)

// melTables are the window and filter bank, computed once.
type melTables struct {
	window  [melFrame]float64
	filters [][melBins]float64 // [frequency bin][mel bin]
}

func newMelTables() *melTables {
	t := &melTables{}
	// Periodic Hann: numpy.hanning(321)[:-1].
	for n := range melFrame {
		t.window[n] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/float64(melFrame))
	}
	// Triangular filters on HTK mel-spaced centres, in hertz.
	hz2mel := func(f float64) float64 { return 2595 * math.Log10(1+f/700) }
	mel2hz := func(m float64) float64 { return 700 * (math.Pow(10, m/2595) - 1) }
	nf := melFFT/2 + 1
	mmax := hz2mel(melMaxHz)
	centres := make([]float64, melBins+2)
	for i := range centres {
		centres[i] = mel2hz(mmax * float64(i) / float64(melBins+1))
	}
	t.filters = make([][melBins]float64, nf)
	for i := range nf {
		f := float64(audioRate/2) * float64(i) / float64(nf-1)
		for j := range melBins {
			down := -(centres[j] - f) / (centres[j+1] - centres[j])
			up := (centres[j+2] - f) / (centres[j+2] - centres[j+1])
			t.filters[i][j] = max(0, min(down, up))
		}
	}
	return t
}

// logMel returns the log-mel features [frames, 128] of 16 kHz samples
// and, per frame, whether it is real audio rather than padding. Padding
// frames are zero, as the extractor leaves them.
func (t *melTables) logMel(samples []float32) ([]float32, []bool) {
	if len(samples) > audioMaxSamples {
		samples = samples[:audioMaxSamples]
	}
	n := len(samples)
	padded := (n + audioPadTo - 1) / audioPadTo * audioPadTo
	// Left padding of half a frame, then frames of melFrame+1 samples are
	// cut every hop and the last sample dropped, as the extractor does.
	const left = melFrame / 2
	total := left + padded
	frames := (total-(melFrame+1))/melHop + 1
	if frames <= 0 {
		return nil, nil
	}
	sample := func(i int) float64 { // index into the left-padded signal
		i -= left
		if i < 0 || i >= n {
			return 0
		}
		return float64(samples[i])
	}
	out := make([]float32, frames*melBins)
	valid := make([]bool, frames)
	buf := make([]complex128, melFFT)
	mag := make([]float64, melFFT/2+1)
	for f := range frames {
		start := f * melHop
		// A frame counts as audio when its last sample (index start+melFrame
		// of the padded signal) is a real sample.
		valid[f] = start+melFrame-left < n
		if !valid[f] {
			continue
		}
		for i := range buf {
			buf[i] = 0
		}
		for i := range melFrame {
			buf[i] = complex(sample(start+i)*t.window[i], 0)
		}
		fft(buf)
		for i := range mag {
			mag[i] = cmplx.Abs(buf[i])
		}
		row := out[f*melBins : (f+1)*melBins]
		for j := range melBins {
			var s float64
			for i, m := range mag {
				s += m * t.filters[i][j]
			}
			row[j] = float32(math.Log(s + melFloor))
		}
	}
	return out, valid
}

// fft is an in-place iterative radix-2 Cooley–Tukey transform; len(x)
// must be a power of two.
func fft(x []complex128) {
	n := len(x)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		w := cmplx.Exp(complex(0, -2*math.Pi/float64(size)))
		for start := 0; start < n; start += size {
			wk := complex(1, 0)
			for k := range size / 2 {
				a, b := x[start+k], x[start+k+size/2]*wk
				x[start+k], x[start+k+size/2] = a+b, a-b
				wk *= w
			}
		}
	}
}

// ReadWAV reads a PCM WAV file with 16-bit samples and returns its
// samples in [−1, 1], channels averaged, and the sample rate.
// EmbedAudio takes 16 kHz; convert other rates before calling it.
func ReadWAV(r io.Reader) ([]float32, int, error) {
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return nil, 0, fmt.Errorf("gemma: WAV header: %w", err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("gemma: not a WAV file")
	}
	var channels, bits uint16
	var rate uint32
	for {
		var hdr [8]byte
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			return nil, 0, fmt.Errorf("gemma: WAV without a data chunk: %w", err)
		}
		id, size := string(hdr[0:4]), binary.LittleEndian.Uint32(hdr[4:8])
		switch id {
		case "fmt ":
			b := make([]byte, size)
			if _, err := io.ReadFull(r, b); err != nil {
				return nil, 0, err
			}
			if format := binary.LittleEndian.Uint16(b[0:2]); format != 1 {
				return nil, 0, fmt.Errorf("gemma: WAV format %d not supported, only 16-bit PCM", format)
			}
			channels = binary.LittleEndian.Uint16(b[2:4])
			rate = binary.LittleEndian.Uint32(b[4:8])
			bits = binary.LittleEndian.Uint16(b[14:16])
		case "data":
			if bits != 16 || channels == 0 {
				return nil, 0, fmt.Errorf("gemma: WAV with %d-bit samples not supported, only 16-bit PCM", bits)
			}
			b := make([]byte, size)
			n, err := io.ReadFull(r, b)
			if err != nil && err != io.ErrUnexpectedEOF {
				return nil, 0, err
			}
			b = b[:n]
			frames := len(b) / 2 / int(channels)
			out := make([]float32, frames)
			for i := range frames {
				var s float32
				for c := range int(channels) {
					s += float32(int16(binary.LittleEndian.Uint16(b[(i*int(channels)+c)*2:]))) / 32768
				}
				out[i] = s / float32(channels)
			}
			return out, int(rate), nil
		default:
			if _, err := io.CopyN(io.Discard, r, int64(size+size%2)); err != nil {
				return nil, 0, err
			}
		}
	}
}
