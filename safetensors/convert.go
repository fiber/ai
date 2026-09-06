package safetensors

import (
	"encoding/binary"
	"math"

	"github.com/fiber/ai/internal/parallel"
)

// Conversions from the stored little-endian bytes into float32. Large
// tensors convert in parallel; the loops are simple enough for the
// compiler to keep them at memory speed.

const convChunk = 1 << 16 // elements per parallel chunk

var converters = map[Dtype]func(dst []float32, src []byte){
	F32:  convertF32,
	BF16: convertBF16,
	F16:  convertF16,
	F64:  convertF64,
}

func convertF32(dst []float32, src []byte) {
	parallel.Range(len(dst), convChunk, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[4*i:]))
		}
	})
}

// convertBF16 is exact: bfloat16 is the top half of a float32.
func convertBF16(dst []float32, src []byte) {
	parallel.Range(len(dst), convChunk, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(src[2*i:])) << 16)
		}
	})
}

func convertF16(dst []float32, src []byte) {
	parallel.Range(len(dst), convChunk, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			dst[i] = f16to32(binary.LittleEndian.Uint16(src[2*i:]))
		}
	})
}

func convertF64(dst []float32, src []byte) {
	parallel.Range(len(dst), convChunk, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			dst[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(src[8*i:])))
		}
	})
}

// f16to32 converts an IEEE 754 binary16 value, including subnormals,
// infinities and NaNs (payload preserved in the top mantissa bits).
func f16to32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: normalise by shifting the mantissa up until the
		// implicit bit appears, adjusting the exponent per shift.
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}
