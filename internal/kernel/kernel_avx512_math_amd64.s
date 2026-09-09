#include "textflag.h"

// AVX-512 transcendental kernels for the avx512 back-ends: the algorithms
// of the AVX2 routines in kernel_avx2_amd64.s (whose constant tables they
// share), sixteen lanes, every constant broadcast once into Z16–Z31 so
// the loop carries no constant loads (the AVX2 versions take twelve
// memory operands per eight elements), two vectors per iteration, mask
// registers for the compares.

// func expAVX512(x, z *float32, n int)     (n % 16 == 0)
TEXT ·expAVX512(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	VBROADCASTSS ·expConsts+0(SB), Z16     // log2e
	VBROADCASTSS ·expConsts+32(SB), Z17    // ln2hi
	VBROADCASTSS ·expConsts+64(SB), Z18    // ln2lo
	VBROADCASTSS ·expConsts+96(SB), Z19    // clamp hi
	VBROADCASTSS ·expConsts+128(SB), Z20   // clamp lo
	VBROADCASTSS ·expConsts+160(SB), Z21   // c0
	VBROADCASTSS ·expConsts+192(SB), Z22   // c1
	VBROADCASTSS ·expConsts+224(SB), Z23   // c2
	VBROADCASTSS ·expConsts+256(SB), Z24   // c3
	VBROADCASTSS ·expConsts+288(SB), Z25   // c4
	VBROADCASTSS ·expConsts+320(SB), Z26   // c5
	VBROADCASTSS ·expConsts+352(SB), Z27   // one
	SHRQ $4, CX
	JZ   expAVX512_done
	MOVQ CX, BX
	SHRQ $1, BX
	JZ   expAVX512_tail
expAVX512_loop2:
	VMOVUPS (SI), Z0
	VMOVUPS 64(SI), Z8
	VCMPPS $5, Z20, Z0, K1                // keep: x >= lo
	VMAXPS Z20, Z0, Z0
	VMINPS Z19, Z0, Z0
	VMULPS Z16, Z0, Z1                    // t = x·log2e
	VRNDSCALEPS $0, Z1, Z1                // n = round(t)
	VCVTPS2DQ Z1, Z2
	VFNMADD231PS Z17, Z1, Z0              // r = x - n·ln2hi
	VFNMADD231PS Z18, Z1, Z0              // r -= n·ln2lo
	VMOVAPS Z21, Z3
	VFMADD213PS Z22, Z0, Z3               // p = c0·r + c1
	VFMADD213PS Z23, Z0, Z3
	VFMADD213PS Z24, Z0, Z3
	VFMADD213PS Z25, Z0, Z3
	VFMADD213PS Z26, Z0, Z3               // p = ... + c5
	VMULPS Z0, Z0, Z4
	VFMADD213PS Z0, Z4, Z3              // p·r² + r
	VADDPS Z27, Z3, Z3                    // + 1
	VPSLLD $23, Z2, Z2
	VPADDD Z2, Z3, Z3                   // p · 2^n via the exponent field
	VPXORD Z5, Z5, Z5
	VMOVAPS Z3, K1, Z5                 // lanes below the clamp stay 0
	VCMPPS $5, Z20, Z8, K2                // keep: x >= lo
	VMAXPS Z20, Z8, Z8
	VMINPS Z19, Z8, Z8
	VMULPS Z16, Z8, Z9                    // t = x·log2e
	VRNDSCALEPS $0, Z9, Z9                // n = round(t)
	VCVTPS2DQ Z9, Z10
	VFNMADD231PS Z17, Z9, Z8              // r = x - n·ln2hi
	VFNMADD231PS Z18, Z9, Z8              // r -= n·ln2lo
	VMOVAPS Z21, Z11
	VFMADD213PS Z22, Z8, Z11               // p = c0·r + c1
	VFMADD213PS Z23, Z8, Z11
	VFMADD213PS Z24, Z8, Z11
	VFMADD213PS Z25, Z8, Z11
	VFMADD213PS Z26, Z8, Z11               // p = ... + c5
	VMULPS Z8, Z8, Z12
	VFMADD213PS Z8, Z12, Z11              // p·r² + r
	VADDPS Z27, Z11, Z11                    // + 1
	VPSLLD $23, Z10, Z10
	VPADDD Z10, Z11, Z11                   // p · 2^n via the exponent field
	VPXORD Z13, Z13, Z13
	VMOVAPS Z11, K2, Z13                 // lanes below the clamp stay 0
	VMOVUPS Z5, (DX)
	VMOVUPS Z13, 64(DX)
	ADDQ $128, SI
	ADDQ $128, DX
	DECQ BX
	JNZ  expAVX512_loop2
	ANDQ $1, CX
	JZ   expAVX512_done
expAVX512_tail:
	VMOVUPS (SI), Z0
	VCMPPS $5, Z20, Z0, K1                // keep: x >= lo
	VMAXPS Z20, Z0, Z0
	VMINPS Z19, Z0, Z0
	VMULPS Z16, Z0, Z1                    // t = x·log2e
	VRNDSCALEPS $0, Z1, Z1                // n = round(t)
	VCVTPS2DQ Z1, Z2
	VFNMADD231PS Z17, Z1, Z0              // r = x - n·ln2hi
	VFNMADD231PS Z18, Z1, Z0              // r -= n·ln2lo
	VMOVAPS Z21, Z3
	VFMADD213PS Z22, Z0, Z3               // p = c0·r + c1
	VFMADD213PS Z23, Z0, Z3
	VFMADD213PS Z24, Z0, Z3
	VFMADD213PS Z25, Z0, Z3
	VFMADD213PS Z26, Z0, Z3               // p = ... + c5
	VMULPS Z0, Z0, Z4
	VFMADD213PS Z0, Z4, Z3              // p·r² + r
	VADDPS Z27, Z3, Z3                    // + 1
	VPSLLD $23, Z2, Z2
	VPADDD Z2, Z3, Z3                   // p · 2^n via the exponent field
	VPXORD Z5, Z5, Z5
	VMOVAPS Z3, K1, Z5                 // lanes below the clamp stay 0
	VMOVUPS Z5, (DX)
expAVX512_done:
	VZEROUPPER
	RET

// func expSumAVX512(x, z *float32, n int, a, b float32) float32   (n % 16 == 0)
//
// z = exp(a·x + b), returns the sum of z; see ExpSum.
TEXT ·expSumAVX512(SB), NOSPLIT, $0-36
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	VBROADCASTSS ·expConsts+0(SB), Z16     // log2e
	VBROADCASTSS ·expConsts+32(SB), Z17    // ln2hi
	VBROADCASTSS ·expConsts+64(SB), Z18    // ln2lo
	VBROADCASTSS ·expConsts+96(SB), Z19    // clamp hi
	VBROADCASTSS ·expConsts+128(SB), Z20   // clamp lo
	VBROADCASTSS ·expConsts+160(SB), Z21   // c0
	VBROADCASTSS ·expConsts+192(SB), Z22   // c1
	VBROADCASTSS ·expConsts+224(SB), Z23   // c2
	VBROADCASTSS ·expConsts+256(SB), Z24   // c3
	VBROADCASTSS ·expConsts+288(SB), Z25   // c4
	VBROADCASTSS ·expConsts+320(SB), Z26   // c5
	VBROADCASTSS ·expConsts+352(SB), Z27   // one
	VBROADCASTSS a+24(FP), Z28
	VBROADCASTSS b+28(FP), Z29
	VPXORD Z30, Z30, Z30                   // two sum accumulators
	VPXORD Z31, Z31, Z31
	SHRQ $4, CX
	JZ   expSumAVX512_done
	MOVQ CX, BX
	SHRQ $1, BX
	JZ   expSumAVX512_tail
expSumAVX512_loop2:
	VMOVUPS (SI), Z0
	VMOVUPS 64(SI), Z8
	VFMADD132PS Z28, Z29, Z0            // a·x + b
	VCMPPS $5, Z20, Z0, K1                // keep: x >= lo
	VMAXPS Z20, Z0, Z0
	VMINPS Z19, Z0, Z0
	VMULPS Z16, Z0, Z1                    // t = x·log2e
	VRNDSCALEPS $0, Z1, Z1                // n = round(t)
	VCVTPS2DQ Z1, Z2
	VFNMADD231PS Z17, Z1, Z0              // r = x - n·ln2hi
	VFNMADD231PS Z18, Z1, Z0              // r -= n·ln2lo
	VMOVAPS Z21, Z3
	VFMADD213PS Z22, Z0, Z3               // p = c0·r + c1
	VFMADD213PS Z23, Z0, Z3
	VFMADD213PS Z24, Z0, Z3
	VFMADD213PS Z25, Z0, Z3
	VFMADD213PS Z26, Z0, Z3               // p = ... + c5
	VMULPS Z0, Z0, Z4
	VFMADD213PS Z0, Z4, Z3              // p·r² + r
	VADDPS Z27, Z3, Z3                    // + 1
	VPSLLD $23, Z2, Z2
	VPADDD Z2, Z3, Z3                   // p · 2^n via the exponent field
	VPXORD Z5, Z5, Z5
	VMOVAPS Z3, K1, Z5                 // lanes below the clamp stay 0
	VADDPS Z5, Z30, Z30
	VFMADD132PS Z28, Z29, Z8            // a·x + b
	VCMPPS $5, Z20, Z8, K2                // keep: x >= lo
	VMAXPS Z20, Z8, Z8
	VMINPS Z19, Z8, Z8
	VMULPS Z16, Z8, Z9                    // t = x·log2e
	VRNDSCALEPS $0, Z9, Z9                // n = round(t)
	VCVTPS2DQ Z9, Z10
	VFNMADD231PS Z17, Z9, Z8              // r = x - n·ln2hi
	VFNMADD231PS Z18, Z9, Z8              // r -= n·ln2lo
	VMOVAPS Z21, Z11
	VFMADD213PS Z22, Z8, Z11               // p = c0·r + c1
	VFMADD213PS Z23, Z8, Z11
	VFMADD213PS Z24, Z8, Z11
	VFMADD213PS Z25, Z8, Z11
	VFMADD213PS Z26, Z8, Z11               // p = ... + c5
	VMULPS Z8, Z8, Z12
	VFMADD213PS Z8, Z12, Z11              // p·r² + r
	VADDPS Z27, Z11, Z11                    // + 1
	VPSLLD $23, Z10, Z10
	VPADDD Z10, Z11, Z11                   // p · 2^n via the exponent field
	VPXORD Z13, Z13, Z13
	VMOVAPS Z11, K2, Z13                 // lanes below the clamp stay 0
	VADDPS Z13, Z31, Z31
	VMOVUPS Z5, (DX)
	VMOVUPS Z13, 64(DX)
	ADDQ $128, SI
	ADDQ $128, DX
	DECQ BX
	JNZ  expSumAVX512_loop2
	ANDQ $1, CX
	JZ   expSumAVX512_done
expSumAVX512_tail:
	VMOVUPS (SI), Z0
	VFMADD132PS Z28, Z29, Z0            // a·x + b
	VCMPPS $5, Z20, Z0, K1                // keep: x >= lo
	VMAXPS Z20, Z0, Z0
	VMINPS Z19, Z0, Z0
	VMULPS Z16, Z0, Z1                    // t = x·log2e
	VRNDSCALEPS $0, Z1, Z1                // n = round(t)
	VCVTPS2DQ Z1, Z2
	VFNMADD231PS Z17, Z1, Z0              // r = x - n·ln2hi
	VFNMADD231PS Z18, Z1, Z0              // r -= n·ln2lo
	VMOVAPS Z21, Z3
	VFMADD213PS Z22, Z0, Z3               // p = c0·r + c1
	VFMADD213PS Z23, Z0, Z3
	VFMADD213PS Z24, Z0, Z3
	VFMADD213PS Z25, Z0, Z3
	VFMADD213PS Z26, Z0, Z3               // p = ... + c5
	VMULPS Z0, Z0, Z4
	VFMADD213PS Z0, Z4, Z3              // p·r² + r
	VADDPS Z27, Z3, Z3                    // + 1
	VPSLLD $23, Z2, Z2
	VPADDD Z2, Z3, Z3                   // p · 2^n via the exponent field
	VPXORD Z5, Z5, Z5
	VMOVAPS Z3, K1, Z5                 // lanes below the clamp stay 0
	VADDPS Z5, Z30, Z30
	VMOVUPS Z5, (DX)
expSumAVX512_done:
	VADDPS Z31, Z30, Z30
	VEXTRACTF64X4 $1, Z30, Y1
	VADDPS Y1, Y30, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+32(FP)
	VZEROUPPER
	RET

// func tanhAVX512(x, z *float32, n int)    (n % 16 == 0)
//
// Rational approximation (see genericTanh); |x| and the sign come from
// broadcast memory operands, the only two loads besides x.
TEXT ·tanhAVX512(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	VBROADCASTSS ·tanhConsts+0(SB), Z16    // clamp
	VBROADCASTSS ·tanhConsts+32(SB), Z17   // -clamp
	VBROADCASTSS ·tanhConsts+64(SB), Z18   // tiny
	VBROADCASTSS ·tanhConsts+96(SB), Z19   // a1
	VBROADCASTSS ·tanhConsts+128(SB), Z20  // a3
	VBROADCASTSS ·tanhConsts+160(SB), Z21  // a5
	VBROADCASTSS ·tanhConsts+192(SB), Z22  // a7
	VBROADCASTSS ·tanhConsts+224(SB), Z23  // a9
	VBROADCASTSS ·tanhConsts+256(SB), Z24  // a11
	VBROADCASTSS ·tanhConsts+288(SB), Z25  // a13
	VBROADCASTSS ·tanhConsts+320(SB), Z26  // b0
	VBROADCASTSS ·tanhConsts+352(SB), Z27  // b2
	VBROADCASTSS ·tanhConsts+384(SB), Z28  // b4
	VBROADCASTSS ·tanhConsts+416(SB), Z29  // b6
	VBROADCASTSS ·tanhConsts+512(SB), Z30  // nine
	VBROADCASTSS ·tanhConsts+544(SB), Z31  // one
	SHRQ $4, CX
	JZ   tanhAVX512_done
	MOVQ CX, BX
	SHRQ $1, BX
	JZ   tanhAVX512_tail
tanhAVX512_loop2:
	VMOVUPS (SI), Z0
	VMOVUPS 64(SI), Z8
	VPANDD.BCST ·tanhConsts+448(SB), Z0, Z4   // |x|
	VPANDD.BCST ·tanhConsts+480(SB), Z0, Z5   // sign bit
	VPORD Z31, Z5, Z5                        // ±1
	VMAXPS Z17, Z0, Z0
	VMINPS Z16, Z0, Z0
	VMULPS Z0, Z0, Z1
	VMOVAPS Z25, Z2                             // a13
	VFMADD213PS Z24, Z1, Z2
	VFMADD213PS Z23, Z1, Z2
	VFMADD213PS Z22, Z1, Z2
	VFMADD213PS Z21, Z1, Z2
	VFMADD213PS Z20, Z1, Z2
	VFMADD213PS Z19, Z1, Z2                   // ... + a1
	VMULPS Z0, Z2, Z2                         // odd numerator
	VMOVAPS Z29, Z3                             // b6
	VFMADD213PS Z28, Z1, Z3
	VFMADD213PS Z27, Z1, Z3
	VFMADD213PS Z26, Z1, Z3                   // ... + b0
	VDIVPS Z3, Z2, Z2
	VCMPPS $1, Z18, Z4, K1                   // |x| < tiny: x itself
	VMOVAPS Z0, K1, Z2
	VCMPPS $5, Z30, Z4, K2                   // |x| >= 9: exactly ±1
	VMOVAPS Z5, K2, Z2
	VPANDD.BCST ·tanhConsts+448(SB), Z8, Z12   // |x|
	VPANDD.BCST ·tanhConsts+480(SB), Z8, Z13   // sign bit
	VPORD Z31, Z13, Z13                        // ±1
	VMAXPS Z17, Z8, Z8
	VMINPS Z16, Z8, Z8
	VMULPS Z8, Z8, Z9
	VMOVAPS Z25, Z10                             // a13
	VFMADD213PS Z24, Z9, Z10
	VFMADD213PS Z23, Z9, Z10
	VFMADD213PS Z22, Z9, Z10
	VFMADD213PS Z21, Z9, Z10
	VFMADD213PS Z20, Z9, Z10
	VFMADD213PS Z19, Z9, Z10                   // ... + a1
	VMULPS Z8, Z10, Z10                         // odd numerator
	VMOVAPS Z29, Z11                             // b6
	VFMADD213PS Z28, Z9, Z11
	VFMADD213PS Z27, Z9, Z11
	VFMADD213PS Z26, Z9, Z11                   // ... + b0
	VDIVPS Z11, Z10, Z10
	VCMPPS $1, Z18, Z12, K3                   // |x| < tiny: x itself
	VMOVAPS Z8, K3, Z10
	VCMPPS $5, Z30, Z12, K4                   // |x| >= 9: exactly ±1
	VMOVAPS Z13, K4, Z10
	VMOVUPS Z2, (DX)
	VMOVUPS Z10, 64(DX)
	ADDQ $128, SI
	ADDQ $128, DX
	DECQ BX
	JNZ  tanhAVX512_loop2
	ANDQ $1, CX
	JZ   tanhAVX512_done
tanhAVX512_tail:
	VMOVUPS (SI), Z0
	VPANDD.BCST ·tanhConsts+448(SB), Z0, Z4   // |x|
	VPANDD.BCST ·tanhConsts+480(SB), Z0, Z5   // sign bit
	VPORD Z31, Z5, Z5                        // ±1
	VMAXPS Z17, Z0, Z0
	VMINPS Z16, Z0, Z0
	VMULPS Z0, Z0, Z1
	VMOVAPS Z25, Z2                             // a13
	VFMADD213PS Z24, Z1, Z2
	VFMADD213PS Z23, Z1, Z2
	VFMADD213PS Z22, Z1, Z2
	VFMADD213PS Z21, Z1, Z2
	VFMADD213PS Z20, Z1, Z2
	VFMADD213PS Z19, Z1, Z2                   // ... + a1
	VMULPS Z0, Z2, Z2                         // odd numerator
	VMOVAPS Z29, Z3                             // b6
	VFMADD213PS Z28, Z1, Z3
	VFMADD213PS Z27, Z1, Z3
	VFMADD213PS Z26, Z1, Z3                   // ... + b0
	VDIVPS Z3, Z2, Z2
	VCMPPS $1, Z18, Z4, K1                   // |x| < tiny: x itself
	VMOVAPS Z0, K1, Z2
	VCMPPS $5, Z30, Z4, K2                   // |x| >= 9: exactly ±1
	VMOVAPS Z5, K2, Z2
	VMOVUPS Z2, (DX)
tanhAVX512_done:
	VZEROUPPER
	RET
