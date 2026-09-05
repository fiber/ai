//go:build amd64

#include "textflag.h"

// float32 AVX2/FMA kernels.
//
// Operand order reminder for Go's AVX syntax: the destination is last and
// Intel's order is reversed, so
//     VFMADD231PS Y12, Y14, Y0   ==   Y0 += Y14 * Y12
//     VSUBPS      (DI), Y0, Y0   ==   Y0  = Y0 - mem
//
// All multi-line macros are defined before the first TEXT because go vet's
// asmdecl attributes macro body lines to the preceding function.

// Binary element-wise kernels: z = x OP y, 32 floats per main iteration,
// then 8, then a scalar tail.
// func xxxAVX2(x, y, z *float32, n int)
#define BINARY_BODY(VOP, SOP) \
	MOVQ x+0(FP), SI \
	MOVQ y+8(FP), DI \
	MOVQ z+16(FP), DX \
	MOVQ n+24(FP), CX \
	MOVQ CX, BX \
	SHRQ $5, BX \
	JZ   tail8 \
loop32: \
	VMOVUPS (SI), Y0 \
	VMOVUPS 32(SI), Y1 \
	VMOVUPS 64(SI), Y2 \
	VMOVUPS 96(SI), Y3 \
	VOP (DI), Y0, Y0 \
	VOP 32(DI), Y1, Y1 \
	VOP 64(DI), Y2, Y2 \
	VOP 96(DI), Y3, Y3 \
	VMOVUPS Y0, (DX) \
	VMOVUPS Y1, 32(DX) \
	VMOVUPS Y2, 64(DX) \
	VMOVUPS Y3, 96(DX) \
	ADDQ $128, SI \
	ADDQ $128, DI \
	ADDQ $128, DX \
	DECQ BX \
	JNZ  loop32 \
tail8: \
	MOVQ CX, BX \
	ANDQ $31, BX \
	SHRQ $3, BX \
	JZ   tail1 \
loop8: \
	VMOVUPS (SI), Y0 \
	VOP (DI), Y0, Y0 \
	VMOVUPS Y0, (DX) \
	ADDQ $32, SI \
	ADDQ $32, DI \
	ADDQ $32, DX \
	DECQ BX \
	JNZ  loop8 \
tail1: \
	ANDQ $7, CX \
	JZ   done \
loop1: \
	VMOVSS (SI), X0 \
	SOP (DI), X0, X0 \
	VMOVSS X0, (DX) \
	ADDQ $4, SI \
	ADDQ $4, DI \
	ADDQ $4, DX \
	DECQ CX \
	JNZ  loop1 \
done: \
	VZEROUPPER \
	RET

// Scalar-broadcast kernels: z = x OP s
// func xxxAVX2(x, z *float32, s float32, n int)
#define SCALAR_BODY(VOP, SOP) \
	MOVQ x+0(FP), SI \
	MOVQ z+8(FP), DX \
	VBROADCASTSS s+16(FP), Y4 \
	MOVQ n+24(FP), CX \
	MOVQ CX, BX \
	SHRQ $5, BX \
	JZ   tail8 \
loop32: \
	VMOVUPS (SI), Y0 \
	VMOVUPS 32(SI), Y1 \
	VMOVUPS 64(SI), Y2 \
	VMOVUPS 96(SI), Y3 \
	VOP Y4, Y0, Y0 \
	VOP Y4, Y1, Y1 \
	VOP Y4, Y2, Y2 \
	VOP Y4, Y3, Y3 \
	VMOVUPS Y0, (DX) \
	VMOVUPS Y1, 32(DX) \
	VMOVUPS Y2, 64(DX) \
	VMOVUPS Y3, 96(DX) \
	ADDQ $128, SI \
	ADDQ $128, DX \
	DECQ BX \
	JNZ  loop32 \
tail8: \
	MOVQ CX, BX \
	ANDQ $31, BX \
	SHRQ $3, BX \
	JZ   tail1 \
loop8: \
	VMOVUPS (SI), Y0 \
	VOP Y4, Y0, Y0 \
	VMOVUPS Y0, (DX) \
	ADDQ $32, SI \
	ADDQ $32, DX \
	DECQ BX \
	JNZ  loop8 \
tail1: \
	ANDQ $7, CX \
	JZ   done \
loop1: \
	VMOVSS (SI), X0 \
	SOP X4, X0, X0 \
	VMOVSS X0, (DX) \
	ADDQ $4, SI \
	ADDQ $4, DX \
	DECQ CX \
	JNZ  loop1 \
done: \
	VZEROUPPER \
	RET

// GEMM micro-kernel k-step (MR=6, NR=16). AOFF/BOFF are byte offsets of the
// current k within the packed panels. Accumulators Y0..Y11 (row r uses
// Y(2r), Y(2r+1)), B in Y12/Y13, broadcast A in Y14/Y15.
#define KSTEP(AOFF, BOFF) \
	PREFETCHT0 (BOFF+512)(BX) \
	PREFETCHT0 (AOFF+192)(AX) \
	VMOVUPS BOFF(BX), Y12 \
	VMOVUPS BOFF+32(BX), Y13 \
	VBROADCASTSS AOFF(AX), Y14 \
	VBROADCASTSS AOFF+4(AX), Y15 \
	VFMADD231PS Y12, Y14, Y0 \
	VFMADD231PS Y13, Y14, Y1 \
	VFMADD231PS Y12, Y15, Y2 \
	VFMADD231PS Y13, Y15, Y3 \
	VBROADCASTSS AOFF+8(AX), Y14 \
	VBROADCASTSS AOFF+12(AX), Y15 \
	VFMADD231PS Y12, Y14, Y4 \
	VFMADD231PS Y13, Y14, Y5 \
	VFMADD231PS Y12, Y15, Y6 \
	VFMADD231PS Y13, Y15, Y7 \
	VBROADCASTSS AOFF+16(AX), Y14 \
	VBROADCASTSS AOFF+20(AX), Y15 \
	VFMADD231PS Y12, Y14, Y8 \
	VFMADD231PS Y13, Y14, Y9 \
	VFMADD231PS Y12, Y15, Y10 \
	VFMADD231PS Y13, Y15, Y11

// CROW(R0, R1): C row += accumulators; advance C by ldc bytes (R8).
#define CROW(R0, R1) \
	VADDPS (DX), R0, R0 \
	VADDPS 32(DX), R1, R1 \
	VMOVUPS R0, (DX) \
	VMOVUPS R1, 32(DX) \
	ADDQ R8, DX

// ---------------------------------------------------------------------------
// Binary element-wise kernels
// ---------------------------------------------------------------------------

TEXT ·addAVX2(SB), NOSPLIT, $0-32
	BINARY_BODY(VADDPS, VADDSS)

TEXT ·subAVX2(SB), NOSPLIT, $0-32
	BINARY_BODY(VSUBPS, VSUBSS)

TEXT ·mulAVX2(SB), NOSPLIT, $0-32
	BINARY_BODY(VMULPS, VMULSS)

TEXT ·divAVX2(SB), NOSPLIT, $0-32
	BINARY_BODY(VDIVPS, VDIVSS)

TEXT ·maximumAVX2(SB), NOSPLIT, $0-32
	BINARY_BODY(VMAXPS, VMAXSS)

// ---------------------------------------------------------------------------
// Scalar-broadcast kernels
// ---------------------------------------------------------------------------

TEXT ·addScalarAVX2(SB), NOSPLIT, $0-32
	SCALAR_BODY(VADDPS, VADDSS)

TEXT ·scaleAVX2(SB), NOSPLIT, $0-32
	SCALAR_BODY(VMULPS, VMULSS)

TEXT ·maxScalarAVX2(SB), NOSPLIT, $0-32
	SCALAR_BODY(VMAXPS, VMAXSS)

// ---------------------------------------------------------------------------
// func axpyAVX2(x, y *float32, alpha float32, n int)   y += alpha * x
// ---------------------------------------------------------------------------
TEXT ·axpyAVX2(SB), NOSPLIT, $0-32
	MOVQ x+0(FP), SI
	MOVQ y+8(FP), DI
	VBROADCASTSS alpha+16(FP), Y4
	MOVQ n+24(FP), CX
	MOVQ CX, BX
	SHRQ $5, BX
	JZ   tail8
loop32:
	VMOVUPS (DI), Y0
	VMOVUPS 32(DI), Y1
	VMOVUPS 64(DI), Y2
	VMOVUPS 96(DI), Y3
	VFMADD231PS (SI), Y4, Y0
	VFMADD231PS 32(SI), Y4, Y1
	VFMADD231PS 64(SI), Y4, Y2
	VFMADD231PS 96(SI), Y4, Y3
	VMOVUPS Y0, (DI)
	VMOVUPS Y1, 32(DI)
	VMOVUPS Y2, 64(DI)
	VMOVUPS Y3, 96(DI)
	ADDQ $128, SI
	ADDQ $128, DI
	DECQ BX
	JNZ  loop32
tail8:
	MOVQ CX, BX
	ANDQ $31, BX
	SHRQ $3, BX
	JZ   tail1
loop8:
	VMOVUPS (DI), Y0
	VFMADD231PS (SI), Y4, Y0
	VMOVUPS Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ BX
	JNZ  loop8
tail1:
	ANDQ $7, CX
	JZ   done
loop1:
	VMOVSS (DI), X0
	VFMADD231SS (SI), X4, X0
	VMOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  loop1
done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func dotAVX2(x, y *float32, n int) float32
// Four independent accumulators; the scalar tail runs after the vector
// reduction because VEX scalar ops zero the upper lanes of the register.
// ---------------------------------------------------------------------------
TEXT ·dotAVX2(SB), NOSPLIT, $0-28
	MOVQ x+0(FP), SI
	MOVQ y+8(FP), DI
	MOVQ n+16(FP), CX
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	MOVQ CX, BX
	SHRQ $5, BX
	JZ   tail8
loop32:
	VMOVUPS (SI), Y4
	VMOVUPS 32(SI), Y5
	VMOVUPS 64(SI), Y6
	VMOVUPS 96(SI), Y7
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y5, Y1
	VFMADD231PS 64(DI), Y6, Y2
	VFMADD231PS 96(DI), Y7, Y3
	ADDQ $128, SI
	ADDQ $128, DI
	DECQ BX
	JNZ  loop32
tail8:
	MOVQ CX, BX
	ANDQ $31, BX
	SHRQ $3, BX
	JZ   reduce
loop8:
	VMOVUPS (SI), Y4
	VFMADD231PS (DI), Y4, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ BX
	JNZ  loop8
reduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	ANDQ $7, CX
	JZ   hsum
loop1:
	VMOVSS (SI), X4
	VFMADD231SS (DI), X4, X0
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  loop1
hsum:
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+24(FP)
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func sumAVX2(x *float32, n int) float32
// ---------------------------------------------------------------------------
TEXT ·sumAVX2(SB), NOSPLIT, $0-20
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	MOVQ CX, BX
	SHRQ $5, BX
	JZ   tail8
loop32:
	VADDPS (SI), Y0, Y0
	VADDPS 32(SI), Y1, Y1
	VADDPS 64(SI), Y2, Y2
	VADDPS 96(SI), Y3, Y3
	ADDQ $128, SI
	DECQ BX
	JNZ  loop32
tail8:
	MOVQ CX, BX
	ANDQ $31, BX
	SHRQ $3, BX
	JZ   reduce
loop8:
	VADDPS (SI), Y0, Y0
	ADDQ $32, SI
	DECQ BX
	JNZ  loop8
reduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	ANDQ $7, CX
	JZ   hsum
loop1:
	VADDSS (SI), X0, X0
	ADDQ $4, SI
	DECQ CX
	JNZ  loop1
hsum:
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+16(FP)
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func maxAVX2(x *float32, n int) float32      (n >= 1)
// ---------------------------------------------------------------------------
TEXT ·maxAVX2(SB), NOSPLIT, $0-20
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	VBROADCASTSS (SI), Y0
	VMOVAPS Y0, Y1
	VMOVAPS Y0, Y2
	VMOVAPS Y0, Y3
	MOVQ CX, BX
	SHRQ $5, BX
	JZ   tail8
loop32:
	VMAXPS (SI), Y0, Y0
	VMAXPS 32(SI), Y1, Y1
	VMAXPS 64(SI), Y2, Y2
	VMAXPS 96(SI), Y3, Y3
	ADDQ $128, SI
	DECQ BX
	JNZ  loop32
tail8:
	MOVQ CX, BX
	ANDQ $31, BX
	SHRQ $3, BX
	JZ   reduce
loop8:
	VMAXPS (SI), Y0, Y0
	ADDQ $32, SI
	DECQ BX
	JNZ  loop8
reduce:
	VMAXPS Y1, Y0, Y0
	VMAXPS Y3, Y2, Y2
	VMAXPS Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VMAXPS X1, X0, X0
	ANDQ $7, CX
	JZ   hmax
loop1:
	VMAXSS (SI), X0, X0
	ADDQ $4, SI
	DECQ CX
	JNZ  loop1
hmax:
	VSHUFPS $0x4E, X0, X0, X1
	VMAXPS X1, X0, X0
	VSHUFPS $0xB1, X0, X0, X1
	VMAXPS X1, X0, X0
	VMOVSS X0, ret+16(FP)
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// GEMM micro-kernel, MR=6, NR=16:  C[6×16] += A[6×k] · B[k×16]
// func gemmAVX2(k int, a, b, c *float32, ldc int)
// A is packed k-major (6 floats per k), B is packed k-major (16 per k).
// ---------------------------------------------------------------------------
TEXT ·gemmAVX2(SB), NOSPLIT, $0-40
	MOVQ k+0(FP), CX
	MOVQ a+8(FP), AX
	MOVQ b+16(FP), BX
	MOVQ c+24(FP), DX
	MOVQ ldc+32(FP), R8
	SHLQ $2, R8
	TESTQ CX, CX
	JZ   done
	// prefetch the six C rows (two lines each) for the store phase
	MOVQ DX, R9
	PREFETCHT0 (R9)
	PREFETCHT0 32(R9)
	ADDQ R8, R9
	PREFETCHT0 (R9)
	PREFETCHT0 32(R9)
	ADDQ R8, R9
	PREFETCHT0 (R9)
	PREFETCHT0 32(R9)
	ADDQ R8, R9
	PREFETCHT0 (R9)
	PREFETCHT0 32(R9)
	ADDQ R8, R9
	PREFETCHT0 (R9)
	PREFETCHT0 32(R9)
	ADDQ R8, R9
	PREFETCHT0 (R9)
	PREFETCHT0 32(R9)
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11
	MOVQ CX, R9
	SHRQ $2, R9
	JZ   ktail
kloop4:
	KSTEP(0, 0)
	KSTEP(24, 64)
	KSTEP(48, 128)
	KSTEP(72, 192)
	ADDQ $96, AX
	ADDQ $256, BX
	DECQ R9
	JNZ  kloop4
ktail:
	ANDQ $3, CX
	JZ   store
kloop1:
	KSTEP(0, 0)
	ADDQ $24, AX
	ADDQ $64, BX
	DECQ CX
	JNZ  kloop1
store:
	CROW(Y0, Y1)
	CROW(Y2, Y3)
	CROW(Y4, Y5)
	CROW(Y6, Y7)
	CROW(Y8, Y9)
	CROW(Y10, Y11)
done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func expAVX2(x, z *float32, n int)      (n % 8 == 0)
//
// Vectorised exp, 8 floats per iteration; see genericExp for the algorithm.
// ---------------------------------------------------------------------------
DATA ·expConsts+0(SB)/4, $0x3FB8AA3B  // log2e
DATA ·expConsts+4(SB)/4, $0x3F318000  // ln2 hi
DATA ·expConsts+8(SB)/4, $0xB95E8083  // ln2 lo
DATA ·expConsts+12(SB)/4, $0x42B0C0A5 // clamp hi
DATA ·expConsts+16(SB)/4, $0xC2AE0000 // clamp lo (-87.0)
DATA ·expConsts+20(SB)/4, $0x39506967 // c0
DATA ·expConsts+24(SB)/4, $0x3AB743CE // c1
DATA ·expConsts+28(SB)/4, $0x3C088908 // c2
DATA ·expConsts+32(SB)/4, $0x3D2AA9C1 // c3
DATA ·expConsts+36(SB)/4, $0x3E2AAAAA // c4
DATA ·expConsts+40(SB)/4, $0x3F000000 // c5
DATA ·expConsts+44(SB)/4, $0x3F800000 // 1.0
GLOBL ·expConsts(SB), RODATA|NOPTR, $48

TEXT ·expAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	VBROADCASTSS ·expConsts+0(SB), Y8   // log2e
	VBROADCASTSS ·expConsts+4(SB), Y9   // ln2 hi
	VBROADCASTSS ·expConsts+8(SB), Y10  // ln2 lo
	VBROADCASTSS ·expConsts+12(SB), Y11 // hi
	VBROADCASTSS ·expConsts+16(SB), Y12 // lo
	VBROADCASTSS ·expConsts+44(SB), Y13 // 1.0
	SHRQ $3, CX
	JZ   done
loop:
	VMOVUPS (SI), Y0
	VMAXPS  Y12, Y0, Y0                 // clamp
	VMINPS  Y11, Y0, Y0
	VMULPS  Y8, Y0, Y1                  // t = x·log2e
	VROUNDPS $0, Y1, Y1                 // n = round-to-nearest-even
	VCVTPS2DQ Y1, Y2                    // n as int32
	VFNMADD231PS Y9, Y1, Y0             // r = x - n·ln2hi
	VFNMADD231PS Y10, Y1, Y0            // r -= n·ln2lo
	VBROADCASTSS ·expConsts+20(SB), Y3  // p = c0
	VBROADCASTSS ·expConsts+24(SB), Y4
	VFMADD213PS Y4, Y0, Y3              // p = p·r + c1
	VBROADCASTSS ·expConsts+28(SB), Y4
	VFMADD213PS Y4, Y0, Y3              // p = p·r + c2
	VBROADCASTSS ·expConsts+32(SB), Y4
	VFMADD213PS Y4, Y0, Y3              // + c3
	VBROADCASTSS ·expConsts+36(SB), Y4
	VFMADD213PS Y4, Y0, Y3              // + c4
	VBROADCASTSS ·expConsts+40(SB), Y4
	VFMADD213PS Y4, Y0, Y3              // + c5
	VMULPS  Y0, Y0, Y4                  // r²
	VFMADD213PS Y0, Y4, Y3              // p = p·r² + r
	VADDPS  Y13, Y3, Y3                 // + 1
	VPSLLD  $23, Y2, Y2                 // n << 23
	VPADDD  Y2, Y3, Y3                  // p · 2^n
	VMOVUPS Y3, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	DECQ CX
	JNZ  loop
done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func tanhAVX2(x, z *float32, n int)     (n % 8 == 0)
//
// Rational approximation (see genericTanh): odd degree-13 numerator over
// even degree-6 denominator of the clamped argument, x itself below 4e-4.
// ---------------------------------------------------------------------------
DATA ·tanhConsts+0(SB)/4, $0x40FFF644 // clamp
DATA ·tanhConsts+4(SB)/4, $0xC0FFF644 // nclamp
DATA ·tanhConsts+8(SB)/4, $0x39D1B717 // tiny
DATA ·tanhConsts+12(SB)/4, $0x3BA059DC // a1
DATA ·tanhConsts+16(SB)/4, $0x3A270DED // a3
DATA ·tanhConsts+20(SB)/4, $0x3779434A // a5
DATA ·tanhConsts+24(SB)/4, $0x335C0041 // a7
DATA ·tanhConsts+28(SB)/4, $0xAEBD37FF // a9
DATA ·tanhConsts+32(SB)/4, $0x2A61337E // a11
DATA ·tanhConsts+36(SB)/4, $0xA59F25C0 // a13
DATA ·tanhConsts+40(SB)/4, $0x3BA059DD // b0
DATA ·tanhConsts+44(SB)/4, $0x3B14AA05 // b2
DATA ·tanhConsts+48(SB)/4, $0x38F895D6 // b4
DATA ·tanhConsts+52(SB)/4, $0x35A0D3D8 // b6
DATA ·tanhConsts+56(SB)/4, $0x7FFFFFFF // abs mask
DATA ·tanhConsts+60(SB)/4, $0x80000000 // sign mask
DATA ·tanhConsts+64(SB)/4, $0x41100000 // 9.0: tanh is exactly ±1 in float32 beyond
DATA ·tanhConsts+68(SB)/4, $0x3F800000 // 1.0
GLOBL ·tanhConsts(SB), RODATA|NOPTR, $72

TEXT ·tanhAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	VBROADCASTSS ·tanhConsts+0(SB), Y8   // clamp
	VBROADCASTSS ·tanhConsts+4(SB), Y9  // -clamp
	VBROADCASTSS ·tanhConsts+8(SB), Y10   // tiny
	VBROADCASTSS ·tanhConsts+56(SB), Y11    // abs mask
	VBROADCASTSS ·tanhConsts+36(SB), Y12
	VBROADCASTSS ·tanhConsts+52(SB), Y13
	SHRQ $3, CX
	JZ   tdone
tloop:
	VMOVUPS (SI), Y0
	VANDPS  Y11, Y0, Y6                  // |x|
	VBROADCASTSS ·tanhConsts+60(SB), Y14
	VANDPS  Y14, Y0, Y14                 // sign bit
	VBROADCASTSS ·tanhConsts+68(SB), Y15
	VORPS   Y15, Y14, Y14                // ±1
	VMAXPS  Y9, Y0, Y0                   // clamp
	VMINPS  Y8, Y0, Y0
	VMULPS  Y0, Y0, Y1                   // x²
	VBROADCASTSS ·tanhConsts+32(SB), Y3
	VMOVAPS Y12, Y2
	VFMADD213PS Y3, Y1, Y2               // p = p·x² + a11
	VBROADCASTSS ·tanhConsts+28(SB), Y3
	VFMADD213PS Y3, Y1, Y2
	VBROADCASTSS ·tanhConsts+24(SB), Y3
	VFMADD213PS Y3, Y1, Y2
	VBROADCASTSS ·tanhConsts+20(SB), Y3
	VFMADD213PS Y3, Y1, Y2
	VBROADCASTSS ·tanhConsts+16(SB), Y3
	VFMADD213PS Y3, Y1, Y2
	VBROADCASTSS ·tanhConsts+12(SB), Y3
	VFMADD213PS Y3, Y1, Y2
	VMULPS  Y0, Y2, Y2                   // p·x
	VBROADCASTSS ·tanhConsts+48(SB), Y3
	VMOVAPS Y13, Y4
	VFMADD213PS Y3, Y1, Y4               // q = q·x² + b4
	VBROADCASTSS ·tanhConsts+44(SB), Y3
	VFMADD213PS Y3, Y1, Y4
	VBROADCASTSS ·tanhConsts+40(SB), Y3
	VFMADD213PS Y3, Y1, Y4
	VDIVPS  Y4, Y2, Y5                   // p / q
	VCMPPS  $1, Y10, Y6, Y7              // |x| < tiny
	VBLENDVPS Y7, Y0, Y5, Y5             // tiny: x itself
	VBROADCASTSS ·tanhConsts+64(SB), Y15
	VCMPPS  $5, Y15, Y6, Y7              // |x| >= 9
	VBLENDVPS Y7, Y14, Y5, Y5            // saturate to ±1
	VMOVUPS Y5, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	DECQ CX
	JNZ  tloop
tdone:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func logAVX2(x, z *float32, n int)      (n % 8 == 0)
//
// Cephes logf (see genericLog): mantissa in [√½, √2) by integer ops,
// degree-9 polynomial, exponent times split ln 2.
// ---------------------------------------------------------------------------
DATA ·logConsts+0(SB)/4, $0x00800000 // minnorm
DATA ·logConsts+4(SB)/4, $0x3F3504F3 // sqrthf
DATA ·logConsts+8(SB)/4, $0x3F800000 // one
DATA ·logConsts+12(SB)/4, $0x3F000000 // half
DATA ·logConsts+16(SB)/4, $0x3D9021BB // p0
DATA ·logConsts+20(SB)/4, $0xBDEBD1B8 // p1
DATA ·logConsts+24(SB)/4, $0x3DEF251A // p2
DATA ·logConsts+28(SB)/4, $0xBDFE5D4F // p3
DATA ·logConsts+32(SB)/4, $0x3E11E9BF // p4
DATA ·logConsts+36(SB)/4, $0xBE2AAE50 // p5
DATA ·logConsts+40(SB)/4, $0x3E4CCEAC // p6
DATA ·logConsts+44(SB)/4, $0xBE7FFFFC // p7
DATA ·logConsts+48(SB)/4, $0x3EAAAAAA // p8
DATA ·logConsts+52(SB)/4, $0xB95E8083 // ln2lo
DATA ·logConsts+56(SB)/4, $0x3F318000 // ln2hi
DATA ·logConsts+60(SB)/4, $0x007FFFFF // mant
DATA ·logConsts+64(SB)/4, $0x0000007E // e126
DATA ·logConsts+68(SB)/4, $0x7F800000 // inf
DATA ·logConsts+72(SB)/4, $0xFF800000 // ninf
DATA ·logConsts+76(SB)/4, $0x7FC00000 // nan
GLOBL ·logConsts(SB), RODATA|NOPTR, $80

TEXT ·logAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	VBROADCASTSS ·logConsts+0(SB), Y8
	VBROADCASTSS ·logConsts+60(SB), Y9
	VBROADCASTSS ·logConsts+12(SB), Y10
	VBROADCASTSS ·logConsts+4(SB), Y11
	VBROADCASTSS ·logConsts+8(SB), Y12
	VBROADCASTSS ·logConsts+64(SB), Y13
	VXORPS  Y14, Y14, Y14                // 0
	SHRQ $3, CX
	JZ   ldone
lloop:
	VMOVUPS (SI), Y0
	VMAXPS  Y8, Y0, Y1                   // xc = max(x, min normal)
	VPSRLD  $23, Y1, Y2                  // exponent field
	VPSUBD  Y13, Y2, Y2                  // e = field - 126
	VCVTDQ2PS Y2, Y2
	VPAND   Y9, Y1, Y3                   // mantissa bits
	VPOR    Y10, Y3, Y3                  // m in [0.5, 1)
	VCMPPS  $1, Y11, Y3, Y4              // m < √½
	VANDPS  Y4, Y3, Y5                   // m or 0
	VANDPS  Y4, Y12, Y6                  // 1 or 0
	VSUBPS  Y6, Y2, Y2                   // e -= 1 where m < √½
	VSUBPS  Y12, Y3, Y3                  // m -= 1
	VADDPS  Y5, Y3, Y3                   // m += m where m < √½
	VMULPS  Y3, Y3, Y7                   // z = m²
	VBROADCASTSS ·logConsts+16(SB), Y15
	VBROADCASTSS ·logConsts+20(SB), Y4
	VFMADD213PS Y4, Y3, Y15              // y = y·m + p1
	VBROADCASTSS ·logConsts+24(SB), Y4
	VFMADD213PS Y4, Y3, Y15
	VBROADCASTSS ·logConsts+28(SB), Y4
	VFMADD213PS Y4, Y3, Y15
	VBROADCASTSS ·logConsts+32(SB), Y4
	VFMADD213PS Y4, Y3, Y15
	VBROADCASTSS ·logConsts+36(SB), Y4
	VFMADD213PS Y4, Y3, Y15
	VBROADCASTSS ·logConsts+40(SB), Y4
	VFMADD213PS Y4, Y3, Y15
	VBROADCASTSS ·logConsts+44(SB), Y4
	VFMADD213PS Y4, Y3, Y15
	VBROADCASTSS ·logConsts+48(SB), Y4
	VFMADD213PS Y4, Y3, Y15
	VMULPS  Y3, Y15, Y15                 // · m
	VMULPS  Y7, Y15, Y15                 // · z
	VBROADCASTSS ·logConsts+52(SB), Y4
	VFMADD231PS Y4, Y2, Y15              // += e·ln2lo
	VFNMADD231PS Y10, Y7, Y15            // -= ½z
	VADDPS  Y3, Y15, Y15                 // + m
	VBROADCASTSS ·logConsts+56(SB), Y4
	VFMADD231PS Y4, Y2, Y15              // += e·ln2hi
	VCMPPS  $0, Y14, Y0, Y4              // x == 0 → -Inf
	VBROADCASTSS ·logConsts+72(SB), Y5
	VBLENDVPS Y4, Y5, Y15, Y15
	VCMPPS  $1, Y14, Y0, Y4              // x < 0 → NaN
	VBROADCASTSS ·logConsts+76(SB), Y5
	VBLENDVPS Y4, Y5, Y15, Y15
	VBROADCASTSS ·logConsts+68(SB), Y5
	VCMPPS  $0, Y5, Y0, Y4               // x == +Inf → +Inf
	VBLENDVPS Y4, Y5, Y15, Y15
	VMOVUPS Y15, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	DECQ CX
	JNZ  lloop
ldone:
	VZEROUPPER
	RET
