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
// CROWZ stores the tile without reading C (β = 0).
#define CROWZ(R0, R1) \
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
// gemmAVX2 accumulates into C, gemmZeroAVX2 overwrites it; both jump into
// the shared body with the mode in R10.
TEXT ·gemmAVX2(SB), NOSPLIT, $0-40
	MOVQ $1, R10
	JMP  ·gemmAVX2Body(SB)

TEXT ·gemmZeroAVX2(SB), NOSPLIT, $0-40
	XORQ R10, R10
	JMP  ·gemmAVX2Body(SB)

TEXT ·gemmAVX2Body(SB), NOSPLIT, $0-40
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
	TESTQ R10, R10
	JZ    storez
	CROW(Y0, Y1)
	CROW(Y2, Y3)
	CROW(Y4, Y5)
	CROW(Y6, Y7)
	CROW(Y8, Y9)
	CROW(Y10, Y11)
	VZEROUPPER
	RET
storez:
	CROWZ(Y0, Y1)
	CROWZ(Y2, Y3)
	CROWZ(Y4, Y5)
	CROWZ(Y6, Y7)
	CROWZ(Y8, Y9)
	CROWZ(Y10, Y11)
done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func expAVX2(x, z *float32, n int)      (n % 8 == 0)
//
// Vectorised exp, see genericExp for the algorithm. Constants are stored
// eight times over so they can be memory operands.
// ---------------------------------------------------------------------------
DATA ·expConsts+0(SB)/4, $0x3FB8AA3B // log2e
DATA ·expConsts+4(SB)/4, $0x3FB8AA3B
DATA ·expConsts+8(SB)/4, $0x3FB8AA3B
DATA ·expConsts+12(SB)/4, $0x3FB8AA3B
DATA ·expConsts+16(SB)/4, $0x3FB8AA3B
DATA ·expConsts+20(SB)/4, $0x3FB8AA3B
DATA ·expConsts+24(SB)/4, $0x3FB8AA3B
DATA ·expConsts+28(SB)/4, $0x3FB8AA3B
DATA ·expConsts+32(SB)/4, $0x3F318000 // ln2hi
DATA ·expConsts+36(SB)/4, $0x3F318000
DATA ·expConsts+40(SB)/4, $0x3F318000
DATA ·expConsts+44(SB)/4, $0x3F318000
DATA ·expConsts+48(SB)/4, $0x3F318000
DATA ·expConsts+52(SB)/4, $0x3F318000
DATA ·expConsts+56(SB)/4, $0x3F318000
DATA ·expConsts+60(SB)/4, $0x3F318000
DATA ·expConsts+64(SB)/4, $0xB95E8083 // ln2lo
DATA ·expConsts+68(SB)/4, $0xB95E8083
DATA ·expConsts+72(SB)/4, $0xB95E8083
DATA ·expConsts+76(SB)/4, $0xB95E8083
DATA ·expConsts+80(SB)/4, $0xB95E8083
DATA ·expConsts+84(SB)/4, $0xB95E8083
DATA ·expConsts+88(SB)/4, $0xB95E8083
DATA ·expConsts+92(SB)/4, $0xB95E8083
DATA ·expConsts+96(SB)/4, $0x42B0C0A5 // hi
DATA ·expConsts+100(SB)/4, $0x42B0C0A5
DATA ·expConsts+104(SB)/4, $0x42B0C0A5
DATA ·expConsts+108(SB)/4, $0x42B0C0A5
DATA ·expConsts+112(SB)/4, $0x42B0C0A5
DATA ·expConsts+116(SB)/4, $0x42B0C0A5
DATA ·expConsts+120(SB)/4, $0x42B0C0A5
DATA ·expConsts+124(SB)/4, $0x42B0C0A5
DATA ·expConsts+128(SB)/4, $0xC2AE0000 // lo
DATA ·expConsts+132(SB)/4, $0xC2AE0000
DATA ·expConsts+136(SB)/4, $0xC2AE0000
DATA ·expConsts+140(SB)/4, $0xC2AE0000
DATA ·expConsts+144(SB)/4, $0xC2AE0000
DATA ·expConsts+148(SB)/4, $0xC2AE0000
DATA ·expConsts+152(SB)/4, $0xC2AE0000
DATA ·expConsts+156(SB)/4, $0xC2AE0000
DATA ·expConsts+160(SB)/4, $0x39506967 // c0
DATA ·expConsts+164(SB)/4, $0x39506967
DATA ·expConsts+168(SB)/4, $0x39506967
DATA ·expConsts+172(SB)/4, $0x39506967
DATA ·expConsts+176(SB)/4, $0x39506967
DATA ·expConsts+180(SB)/4, $0x39506967
DATA ·expConsts+184(SB)/4, $0x39506967
DATA ·expConsts+188(SB)/4, $0x39506967
DATA ·expConsts+192(SB)/4, $0x3AB743CE // c1
DATA ·expConsts+196(SB)/4, $0x3AB743CE
DATA ·expConsts+200(SB)/4, $0x3AB743CE
DATA ·expConsts+204(SB)/4, $0x3AB743CE
DATA ·expConsts+208(SB)/4, $0x3AB743CE
DATA ·expConsts+212(SB)/4, $0x3AB743CE
DATA ·expConsts+216(SB)/4, $0x3AB743CE
DATA ·expConsts+220(SB)/4, $0x3AB743CE
DATA ·expConsts+224(SB)/4, $0x3C088908 // c2
DATA ·expConsts+228(SB)/4, $0x3C088908
DATA ·expConsts+232(SB)/4, $0x3C088908
DATA ·expConsts+236(SB)/4, $0x3C088908
DATA ·expConsts+240(SB)/4, $0x3C088908
DATA ·expConsts+244(SB)/4, $0x3C088908
DATA ·expConsts+248(SB)/4, $0x3C088908
DATA ·expConsts+252(SB)/4, $0x3C088908
DATA ·expConsts+256(SB)/4, $0x3D2AA9C1 // c3
DATA ·expConsts+260(SB)/4, $0x3D2AA9C1
DATA ·expConsts+264(SB)/4, $0x3D2AA9C1
DATA ·expConsts+268(SB)/4, $0x3D2AA9C1
DATA ·expConsts+272(SB)/4, $0x3D2AA9C1
DATA ·expConsts+276(SB)/4, $0x3D2AA9C1
DATA ·expConsts+280(SB)/4, $0x3D2AA9C1
DATA ·expConsts+284(SB)/4, $0x3D2AA9C1
DATA ·expConsts+288(SB)/4, $0x3E2AAAAA // c4
DATA ·expConsts+292(SB)/4, $0x3E2AAAAA
DATA ·expConsts+296(SB)/4, $0x3E2AAAAA
DATA ·expConsts+300(SB)/4, $0x3E2AAAAA
DATA ·expConsts+304(SB)/4, $0x3E2AAAAA
DATA ·expConsts+308(SB)/4, $0x3E2AAAAA
DATA ·expConsts+312(SB)/4, $0x3E2AAAAA
DATA ·expConsts+316(SB)/4, $0x3E2AAAAA
DATA ·expConsts+320(SB)/4, $0x3F000000 // c5
DATA ·expConsts+324(SB)/4, $0x3F000000
DATA ·expConsts+328(SB)/4, $0x3F000000
DATA ·expConsts+332(SB)/4, $0x3F000000
DATA ·expConsts+336(SB)/4, $0x3F000000
DATA ·expConsts+340(SB)/4, $0x3F000000
DATA ·expConsts+344(SB)/4, $0x3F000000
DATA ·expConsts+348(SB)/4, $0x3F000000
DATA ·expConsts+352(SB)/4, $0x3F800000 // one
DATA ·expConsts+356(SB)/4, $0x3F800000
DATA ·expConsts+360(SB)/4, $0x3F800000
DATA ·expConsts+364(SB)/4, $0x3F800000
DATA ·expConsts+368(SB)/4, $0x3F800000
DATA ·expConsts+372(SB)/4, $0x3F800000
DATA ·expConsts+376(SB)/4, $0x3F800000
DATA ·expConsts+380(SB)/4, $0x3F800000
GLOBL ·expConsts(SB), RODATA|NOPTR, $384

TEXT ·expAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	SHRQ $3, CX
	JZ   expAVX2_done
	// Two vectors per iteration with independent registers: consecutive
	// iterations of the single-vector loop did not overlap (1.7 ns per
	// element on Skylake-SP, 1.4 on Apple M2 for the NEON version).
	MOVQ CX, BX
	SHRQ $1, BX
	JZ   expAVX2_tail
expAVX2_loop2:
	VMOVUPS (SI), Y0
	VMOVUPS 32(SI), Y5
	// Lanes below the low clamp return exactly 0 (see genericExp): the
	// mask is taken before clamping and applied before the store.
	VCMPPS $1, ·expConsts+128(SB), Y0, Y10
	VCMPPS $1, ·expConsts+128(SB), Y5, Y11
	VMAXPS ·expConsts+128(SB), Y0, Y0
	VMAXPS ·expConsts+128(SB), Y5, Y5
	VMINPS ·expConsts+96(SB), Y0, Y0
	VMINPS ·expConsts+96(SB), Y5, Y5
	VMULPS ·expConsts+0(SB), Y0, Y1
	VMULPS ·expConsts+0(SB), Y5, Y6
	VROUNDPS $0, Y1, Y1
	VROUNDPS $0, Y6, Y6
	VCVTPS2DQ Y1, Y2
	VCVTPS2DQ Y6, Y7
	VFNMADD231PS ·expConsts+32(SB), Y1, Y0
	VFNMADD231PS ·expConsts+32(SB), Y6, Y5
	VFNMADD231PS ·expConsts+64(SB), Y1, Y0
	VFNMADD231PS ·expConsts+64(SB), Y6, Y5
	VMOVUPS ·expConsts+160(SB), Y3
	VMOVUPS ·expConsts+160(SB), Y8
	VFMADD213PS ·expConsts+192(SB), Y0, Y3
	VFMADD213PS ·expConsts+192(SB), Y5, Y8
	VFMADD213PS ·expConsts+224(SB), Y0, Y3
	VFMADD213PS ·expConsts+224(SB), Y5, Y8
	VFMADD213PS ·expConsts+256(SB), Y0, Y3
	VFMADD213PS ·expConsts+256(SB), Y5, Y8
	VFMADD213PS ·expConsts+288(SB), Y0, Y3
	VFMADD213PS ·expConsts+288(SB), Y5, Y8
	VFMADD213PS ·expConsts+320(SB), Y0, Y3
	VFMADD213PS ·expConsts+320(SB), Y5, Y8
	VMULPS Y0, Y0, Y4
	VMULPS Y5, Y5, Y9
	VFMADD213PS Y0, Y4, Y3
	VFMADD213PS Y5, Y9, Y8
	VADDPS ·expConsts+352(SB), Y3, Y3
	VADDPS ·expConsts+352(SB), Y8, Y8
	VPSLLD $23, Y2, Y2
	VPSLLD $23, Y7, Y7
	VPADDD Y2, Y3, Y3
	VPADDD Y7, Y8, Y8
	VANDNPS Y3, Y10, Y3
	VANDNPS Y8, Y11, Y8
	VMOVUPS Y3, (DX)
	VMOVUPS Y8, 32(DX)
	ADDQ $64, SI
	ADDQ $64, DX
	DECQ BX
	JNZ  expAVX2_loop2
	ANDQ $1, CX
	JZ   expAVX2_done
expAVX2_tail:
	VMOVUPS (SI), Y0
	VCMPPS $1, ·expConsts+128(SB), Y0, Y10
	VMAXPS ·expConsts+128(SB), Y0, Y0
	VMINPS ·expConsts+96(SB), Y0, Y0
	VMULPS ·expConsts+0(SB), Y0, Y1
	VROUNDPS $0, Y1, Y1
	VCVTPS2DQ Y1, Y2
	VFNMADD231PS ·expConsts+32(SB), Y1, Y0
	VFNMADD231PS ·expConsts+64(SB), Y1, Y0
	VMOVUPS ·expConsts+160(SB), Y3
	VFMADD213PS ·expConsts+192(SB), Y0, Y3
	VFMADD213PS ·expConsts+224(SB), Y0, Y3
	VFMADD213PS ·expConsts+256(SB), Y0, Y3
	VFMADD213PS ·expConsts+288(SB), Y0, Y3
	VFMADD213PS ·expConsts+320(SB), Y0, Y3
	VMULPS Y0, Y0, Y4
	VFMADD213PS Y0, Y4, Y3
	VADDPS ·expConsts+352(SB), Y3, Y3
	VPSLLD $23, Y2, Y2
	VPADDD Y2, Y3, Y3
	VANDNPS Y3, Y10, Y3
	VMOVUPS Y3, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	DECQ CX
	JNZ  expAVX2_tail
expAVX2_done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func expSumAVX2(x, z *float32, n int, a, b float32) float32   (n % 8 == 0)
//
// expAVX2 of a·x + b with the stored results accumulated in Y12/Y13 and reduced at
// the end: the softmax normaliser from the same pass.
// ---------------------------------------------------------------------------
TEXT ·expSumAVX2(SB), NOSPLIT, $0-36
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	VXORPS Y12, Y12, Y12 // two sum accumulators
	VXORPS Y13, Y13, Y13
	VBROADCASTSS a+24(FP), Y14
	VBROADCASTSS b+28(FP), Y15
	SHRQ $3, CX
	JZ   expSumAVX2_done
	// Two vectors per iteration with independent registers: consecutive
	// iterations of the single-vector loop did not overlap (1.7 ns per
	// element on Skylake-SP, 1.4 on Apple M2 for the NEON version).
	MOVQ CX, BX
	SHRQ $1, BX
	JZ   expSumAVX2_tail
expSumAVX2_loop2:
	VMOVUPS (SI), Y0
	VFMADD132PS Y14, Y15, Y0 // a·x + b
	VMOVUPS 32(SI), Y5
	VFMADD132PS Y14, Y15, Y5
	// Lanes below the low clamp return exactly 0 (see genericExp): the
	// mask is taken before clamping and applied before the store.
	VCMPPS $1, ·expConsts+128(SB), Y0, Y10
	VCMPPS $1, ·expConsts+128(SB), Y5, Y11
	VMAXPS ·expConsts+128(SB), Y0, Y0
	VMAXPS ·expConsts+128(SB), Y5, Y5
	VMINPS ·expConsts+96(SB), Y0, Y0
	VMINPS ·expConsts+96(SB), Y5, Y5
	VMULPS ·expConsts+0(SB), Y0, Y1
	VMULPS ·expConsts+0(SB), Y5, Y6
	VROUNDPS $0, Y1, Y1
	VROUNDPS $0, Y6, Y6
	VCVTPS2DQ Y1, Y2
	VCVTPS2DQ Y6, Y7
	VFNMADD231PS ·expConsts+32(SB), Y1, Y0
	VFNMADD231PS ·expConsts+32(SB), Y6, Y5
	VFNMADD231PS ·expConsts+64(SB), Y1, Y0
	VFNMADD231PS ·expConsts+64(SB), Y6, Y5
	VMOVUPS ·expConsts+160(SB), Y3
	VMOVUPS ·expConsts+160(SB), Y8
	VFMADD213PS ·expConsts+192(SB), Y0, Y3
	VFMADD213PS ·expConsts+192(SB), Y5, Y8
	VFMADD213PS ·expConsts+224(SB), Y0, Y3
	VFMADD213PS ·expConsts+224(SB), Y5, Y8
	VFMADD213PS ·expConsts+256(SB), Y0, Y3
	VFMADD213PS ·expConsts+256(SB), Y5, Y8
	VFMADD213PS ·expConsts+288(SB), Y0, Y3
	VFMADD213PS ·expConsts+288(SB), Y5, Y8
	VFMADD213PS ·expConsts+320(SB), Y0, Y3
	VFMADD213PS ·expConsts+320(SB), Y5, Y8
	VMULPS Y0, Y0, Y4
	VMULPS Y5, Y5, Y9
	VFMADD213PS Y0, Y4, Y3
	VFMADD213PS Y5, Y9, Y8
	VADDPS ·expConsts+352(SB), Y3, Y3
	VADDPS ·expConsts+352(SB), Y8, Y8
	VPSLLD $23, Y2, Y2
	VPSLLD $23, Y7, Y7
	VPADDD Y2, Y3, Y3
	VPADDD Y7, Y8, Y8
	VANDNPS Y3, Y10, Y3
	VANDNPS Y8, Y11, Y8
	VMOVUPS Y3, (DX)
	VADDPS Y3, Y12, Y12
	VMOVUPS Y8, 32(DX)
	VADDPS Y8, Y13, Y13
	ADDQ $64, SI
	ADDQ $64, DX
	DECQ BX
	JNZ  expSumAVX2_loop2
	ANDQ $1, CX
	JZ   expSumAVX2_done
expSumAVX2_tail:
	VMOVUPS (SI), Y0
	VFMADD132PS Y14, Y15, Y0 // a·x + b
	VCMPPS $1, ·expConsts+128(SB), Y0, Y10
	VMAXPS ·expConsts+128(SB), Y0, Y0
	VMINPS ·expConsts+96(SB), Y0, Y0
	VMULPS ·expConsts+0(SB), Y0, Y1
	VROUNDPS $0, Y1, Y1
	VCVTPS2DQ Y1, Y2
	VFNMADD231PS ·expConsts+32(SB), Y1, Y0
	VFNMADD231PS ·expConsts+64(SB), Y1, Y0
	VMOVUPS ·expConsts+160(SB), Y3
	VFMADD213PS ·expConsts+192(SB), Y0, Y3
	VFMADD213PS ·expConsts+224(SB), Y0, Y3
	VFMADD213PS ·expConsts+256(SB), Y0, Y3
	VFMADD213PS ·expConsts+288(SB), Y0, Y3
	VFMADD213PS ·expConsts+320(SB), Y0, Y3
	VMULPS Y0, Y0, Y4
	VFMADD213PS Y0, Y4, Y3
	VADDPS ·expConsts+352(SB), Y3, Y3
	VPSLLD $23, Y2, Y2
	VPADDD Y2, Y3, Y3
	VANDNPS Y3, Y10, Y3
	VMOVUPS Y3, (DX)
	VADDPS Y3, Y12, Y12
	ADDQ $32, SI
	ADDQ $32, DX
	DECQ CX
	JNZ  expSumAVX2_tail
expSumAVX2_done:
	VADDPS Y13, Y12, Y12
	VEXTRACTF128 $1, Y12, X13
	VADDPS X13, X12, X12
	VHADDPS X12, X12, X12
	VHADDPS X12, X12, X12
	VMOVSS X12, ret+32(FP)
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func tanhAVX2(x, z *float32, n int)     (n % 8 == 0)
//
// Rational approximation (see genericTanh): odd degree-13 numerator over
// even degree-6 denominator of the clamped argument, x itself below 4e-4,
// exactly ±1 from |x| >= 9.
// ---------------------------------------------------------------------------
DATA ·tanhConsts+0(SB)/4, $0x40FFF644 // clamp
DATA ·tanhConsts+4(SB)/4, $0x40FFF644
DATA ·tanhConsts+8(SB)/4, $0x40FFF644
DATA ·tanhConsts+12(SB)/4, $0x40FFF644
DATA ·tanhConsts+16(SB)/4, $0x40FFF644
DATA ·tanhConsts+20(SB)/4, $0x40FFF644
DATA ·tanhConsts+24(SB)/4, $0x40FFF644
DATA ·tanhConsts+28(SB)/4, $0x40FFF644
DATA ·tanhConsts+32(SB)/4, $0xC0FFF644 // nclamp
DATA ·tanhConsts+36(SB)/4, $0xC0FFF644
DATA ·tanhConsts+40(SB)/4, $0xC0FFF644
DATA ·tanhConsts+44(SB)/4, $0xC0FFF644
DATA ·tanhConsts+48(SB)/4, $0xC0FFF644
DATA ·tanhConsts+52(SB)/4, $0xC0FFF644
DATA ·tanhConsts+56(SB)/4, $0xC0FFF644
DATA ·tanhConsts+60(SB)/4, $0xC0FFF644
DATA ·tanhConsts+64(SB)/4, $0x39D1B717 // tiny
DATA ·tanhConsts+68(SB)/4, $0x39D1B717
DATA ·tanhConsts+72(SB)/4, $0x39D1B717
DATA ·tanhConsts+76(SB)/4, $0x39D1B717
DATA ·tanhConsts+80(SB)/4, $0x39D1B717
DATA ·tanhConsts+84(SB)/4, $0x39D1B717
DATA ·tanhConsts+88(SB)/4, $0x39D1B717
DATA ·tanhConsts+92(SB)/4, $0x39D1B717
DATA ·tanhConsts+96(SB)/4, $0x3BA059DC // a1
DATA ·tanhConsts+100(SB)/4, $0x3BA059DC
DATA ·tanhConsts+104(SB)/4, $0x3BA059DC
DATA ·tanhConsts+108(SB)/4, $0x3BA059DC
DATA ·tanhConsts+112(SB)/4, $0x3BA059DC
DATA ·tanhConsts+116(SB)/4, $0x3BA059DC
DATA ·tanhConsts+120(SB)/4, $0x3BA059DC
DATA ·tanhConsts+124(SB)/4, $0x3BA059DC
DATA ·tanhConsts+128(SB)/4, $0x3A270DED // a3
DATA ·tanhConsts+132(SB)/4, $0x3A270DED
DATA ·tanhConsts+136(SB)/4, $0x3A270DED
DATA ·tanhConsts+140(SB)/4, $0x3A270DED
DATA ·tanhConsts+144(SB)/4, $0x3A270DED
DATA ·tanhConsts+148(SB)/4, $0x3A270DED
DATA ·tanhConsts+152(SB)/4, $0x3A270DED
DATA ·tanhConsts+156(SB)/4, $0x3A270DED
DATA ·tanhConsts+160(SB)/4, $0x3779434A // a5
DATA ·tanhConsts+164(SB)/4, $0x3779434A
DATA ·tanhConsts+168(SB)/4, $0x3779434A
DATA ·tanhConsts+172(SB)/4, $0x3779434A
DATA ·tanhConsts+176(SB)/4, $0x3779434A
DATA ·tanhConsts+180(SB)/4, $0x3779434A
DATA ·tanhConsts+184(SB)/4, $0x3779434A
DATA ·tanhConsts+188(SB)/4, $0x3779434A
DATA ·tanhConsts+192(SB)/4, $0x335C0041 // a7
DATA ·tanhConsts+196(SB)/4, $0x335C0041
DATA ·tanhConsts+200(SB)/4, $0x335C0041
DATA ·tanhConsts+204(SB)/4, $0x335C0041
DATA ·tanhConsts+208(SB)/4, $0x335C0041
DATA ·tanhConsts+212(SB)/4, $0x335C0041
DATA ·tanhConsts+216(SB)/4, $0x335C0041
DATA ·tanhConsts+220(SB)/4, $0x335C0041
DATA ·tanhConsts+224(SB)/4, $0xAEBD37FF // a9
DATA ·tanhConsts+228(SB)/4, $0xAEBD37FF
DATA ·tanhConsts+232(SB)/4, $0xAEBD37FF
DATA ·tanhConsts+236(SB)/4, $0xAEBD37FF
DATA ·tanhConsts+240(SB)/4, $0xAEBD37FF
DATA ·tanhConsts+244(SB)/4, $0xAEBD37FF
DATA ·tanhConsts+248(SB)/4, $0xAEBD37FF
DATA ·tanhConsts+252(SB)/4, $0xAEBD37FF
DATA ·tanhConsts+256(SB)/4, $0x2A61337E // a11
DATA ·tanhConsts+260(SB)/4, $0x2A61337E
DATA ·tanhConsts+264(SB)/4, $0x2A61337E
DATA ·tanhConsts+268(SB)/4, $0x2A61337E
DATA ·tanhConsts+272(SB)/4, $0x2A61337E
DATA ·tanhConsts+276(SB)/4, $0x2A61337E
DATA ·tanhConsts+280(SB)/4, $0x2A61337E
DATA ·tanhConsts+284(SB)/4, $0x2A61337E
DATA ·tanhConsts+288(SB)/4, $0xA59F25C0 // a13
DATA ·tanhConsts+292(SB)/4, $0xA59F25C0
DATA ·tanhConsts+296(SB)/4, $0xA59F25C0
DATA ·tanhConsts+300(SB)/4, $0xA59F25C0
DATA ·tanhConsts+304(SB)/4, $0xA59F25C0
DATA ·tanhConsts+308(SB)/4, $0xA59F25C0
DATA ·tanhConsts+312(SB)/4, $0xA59F25C0
DATA ·tanhConsts+316(SB)/4, $0xA59F25C0
DATA ·tanhConsts+320(SB)/4, $0x3BA059DD // b0
DATA ·tanhConsts+324(SB)/4, $0x3BA059DD
DATA ·tanhConsts+328(SB)/4, $0x3BA059DD
DATA ·tanhConsts+332(SB)/4, $0x3BA059DD
DATA ·tanhConsts+336(SB)/4, $0x3BA059DD
DATA ·tanhConsts+340(SB)/4, $0x3BA059DD
DATA ·tanhConsts+344(SB)/4, $0x3BA059DD
DATA ·tanhConsts+348(SB)/4, $0x3BA059DD
DATA ·tanhConsts+352(SB)/4, $0x3B14AA05 // b2
DATA ·tanhConsts+356(SB)/4, $0x3B14AA05
DATA ·tanhConsts+360(SB)/4, $0x3B14AA05
DATA ·tanhConsts+364(SB)/4, $0x3B14AA05
DATA ·tanhConsts+368(SB)/4, $0x3B14AA05
DATA ·tanhConsts+372(SB)/4, $0x3B14AA05
DATA ·tanhConsts+376(SB)/4, $0x3B14AA05
DATA ·tanhConsts+380(SB)/4, $0x3B14AA05
DATA ·tanhConsts+384(SB)/4, $0x38F895D6 // b4
DATA ·tanhConsts+388(SB)/4, $0x38F895D6
DATA ·tanhConsts+392(SB)/4, $0x38F895D6
DATA ·tanhConsts+396(SB)/4, $0x38F895D6
DATA ·tanhConsts+400(SB)/4, $0x38F895D6
DATA ·tanhConsts+404(SB)/4, $0x38F895D6
DATA ·tanhConsts+408(SB)/4, $0x38F895D6
DATA ·tanhConsts+412(SB)/4, $0x38F895D6
DATA ·tanhConsts+416(SB)/4, $0x35A0D3D8 // b6
DATA ·tanhConsts+420(SB)/4, $0x35A0D3D8
DATA ·tanhConsts+424(SB)/4, $0x35A0D3D8
DATA ·tanhConsts+428(SB)/4, $0x35A0D3D8
DATA ·tanhConsts+432(SB)/4, $0x35A0D3D8
DATA ·tanhConsts+436(SB)/4, $0x35A0D3D8
DATA ·tanhConsts+440(SB)/4, $0x35A0D3D8
DATA ·tanhConsts+444(SB)/4, $0x35A0D3D8
DATA ·tanhConsts+448(SB)/4, $0x7FFFFFFF // abs
DATA ·tanhConsts+452(SB)/4, $0x7FFFFFFF
DATA ·tanhConsts+456(SB)/4, $0x7FFFFFFF
DATA ·tanhConsts+460(SB)/4, $0x7FFFFFFF
DATA ·tanhConsts+464(SB)/4, $0x7FFFFFFF
DATA ·tanhConsts+468(SB)/4, $0x7FFFFFFF
DATA ·tanhConsts+472(SB)/4, $0x7FFFFFFF
DATA ·tanhConsts+476(SB)/4, $0x7FFFFFFF
DATA ·tanhConsts+480(SB)/4, $0x80000000 // sign
DATA ·tanhConsts+484(SB)/4, $0x80000000
DATA ·tanhConsts+488(SB)/4, $0x80000000
DATA ·tanhConsts+492(SB)/4, $0x80000000
DATA ·tanhConsts+496(SB)/4, $0x80000000
DATA ·tanhConsts+500(SB)/4, $0x80000000
DATA ·tanhConsts+504(SB)/4, $0x80000000
DATA ·tanhConsts+508(SB)/4, $0x80000000
DATA ·tanhConsts+512(SB)/4, $0x41100000 // nine
DATA ·tanhConsts+516(SB)/4, $0x41100000
DATA ·tanhConsts+520(SB)/4, $0x41100000
DATA ·tanhConsts+524(SB)/4, $0x41100000
DATA ·tanhConsts+528(SB)/4, $0x41100000
DATA ·tanhConsts+532(SB)/4, $0x41100000
DATA ·tanhConsts+536(SB)/4, $0x41100000
DATA ·tanhConsts+540(SB)/4, $0x41100000
DATA ·tanhConsts+544(SB)/4, $0x3F800000 // one
DATA ·tanhConsts+548(SB)/4, $0x3F800000
DATA ·tanhConsts+552(SB)/4, $0x3F800000
DATA ·tanhConsts+556(SB)/4, $0x3F800000
DATA ·tanhConsts+560(SB)/4, $0x3F800000
DATA ·tanhConsts+564(SB)/4, $0x3F800000
DATA ·tanhConsts+568(SB)/4, $0x3F800000
DATA ·tanhConsts+572(SB)/4, $0x3F800000
GLOBL ·tanhConsts(SB), RODATA|NOPTR, $576

TEXT ·tanhAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	SHRQ $3, CX
	JZ   tanhAVX2_done
	// Two vectors per iteration with independent registers: consecutive
	// iterations of the single-vector loop did not overlap (1.7 ns per
	// element on Skylake-SP, 1.4 on Apple M2 for the NEON version).
	MOVQ CX, BX
	SHRQ $1, BX
	JZ   tanhAVX2_tail
tanhAVX2_loop2:
	VMOVUPS (SI), Y0
	VMOVUPS 32(SI), Y7
	VANDPS ·tanhConsts+448(SB), Y0, Y4
	VANDPS ·tanhConsts+448(SB), Y7, Y11
	VANDPS ·tanhConsts+480(SB), Y0, Y5
	VANDPS ·tanhConsts+480(SB), Y7, Y12
	VORPS ·tanhConsts+544(SB), Y5, Y5
	VORPS ·tanhConsts+544(SB), Y12, Y12
	VMAXPS ·tanhConsts+32(SB), Y0, Y0
	VMAXPS ·tanhConsts+32(SB), Y7, Y7
	VMINPS ·tanhConsts+0(SB), Y0, Y0
	VMINPS ·tanhConsts+0(SB), Y7, Y7
	VMULPS Y0, Y0, Y1
	VMULPS Y7, Y7, Y8
	VMOVUPS ·tanhConsts+288(SB), Y2
	VMOVUPS ·tanhConsts+288(SB), Y9
	VFMADD213PS ·tanhConsts+256(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+256(SB), Y8, Y9
	VFMADD213PS ·tanhConsts+224(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+224(SB), Y8, Y9
	VFMADD213PS ·tanhConsts+192(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+192(SB), Y8, Y9
	VFMADD213PS ·tanhConsts+160(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+160(SB), Y8, Y9
	VFMADD213PS ·tanhConsts+128(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+128(SB), Y8, Y9
	VFMADD213PS ·tanhConsts+96(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+96(SB), Y8, Y9
	VMULPS Y0, Y2, Y2
	VMULPS Y7, Y9, Y9
	VMOVUPS ·tanhConsts+416(SB), Y3
	VMOVUPS ·tanhConsts+416(SB), Y10
	VFMADD213PS ·tanhConsts+384(SB), Y1, Y3
	VFMADD213PS ·tanhConsts+384(SB), Y8, Y10
	VFMADD213PS ·tanhConsts+352(SB), Y1, Y3
	VFMADD213PS ·tanhConsts+352(SB), Y8, Y10
	VFMADD213PS ·tanhConsts+320(SB), Y1, Y3
	VFMADD213PS ·tanhConsts+320(SB), Y8, Y10
	VDIVPS Y3, Y2, Y2
	VDIVPS Y10, Y9, Y9
	VCMPPS $1, ·tanhConsts+64(SB), Y4, Y6
	VCMPPS $1, ·tanhConsts+64(SB), Y11, Y13
	VBLENDVPS Y6, Y0, Y2, Y2
	VBLENDVPS Y13, Y7, Y9, Y9
	VCMPPS $5, ·tanhConsts+512(SB), Y4, Y6
	VCMPPS $5, ·tanhConsts+512(SB), Y11, Y13
	VBLENDVPS Y6, Y5, Y2, Y2
	VBLENDVPS Y13, Y12, Y9, Y9
	VMOVUPS Y2, (DX)
	VMOVUPS Y9, 32(DX)
	ADDQ $64, SI
	ADDQ $64, DX
	DECQ BX
	JNZ  tanhAVX2_loop2
	ANDQ $1, CX
	JZ   tanhAVX2_done
tanhAVX2_tail:
	VMOVUPS (SI), Y0
	VANDPS ·tanhConsts+448(SB), Y0, Y4
	VANDPS ·tanhConsts+480(SB), Y0, Y5
	VORPS ·tanhConsts+544(SB), Y5, Y5
	VMAXPS ·tanhConsts+32(SB), Y0, Y0
	VMINPS ·tanhConsts+0(SB), Y0, Y0
	VMULPS Y0, Y0, Y1
	VMOVUPS ·tanhConsts+288(SB), Y2
	VFMADD213PS ·tanhConsts+256(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+224(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+192(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+160(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+128(SB), Y1, Y2
	VFMADD213PS ·tanhConsts+96(SB), Y1, Y2
	VMULPS Y0, Y2, Y2
	VMOVUPS ·tanhConsts+416(SB), Y3
	VFMADD213PS ·tanhConsts+384(SB), Y1, Y3
	VFMADD213PS ·tanhConsts+352(SB), Y1, Y3
	VFMADD213PS ·tanhConsts+320(SB), Y1, Y3
	VDIVPS Y3, Y2, Y2
	VCMPPS $1, ·tanhConsts+64(SB), Y4, Y6
	VBLENDVPS Y6, Y0, Y2, Y2
	VCMPPS $5, ·tanhConsts+512(SB), Y4, Y6
	VBLENDVPS Y6, Y5, Y2, Y2
	VMOVUPS Y2, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	DECQ CX
	JNZ  tanhAVX2_tail
tanhAVX2_done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func logAVX2(x, z *float32, n int)      (n % 8 == 0)
//
// Cephes logf (see genericLog): mantissa in [√½, √2) by integer ops,
// degree-9 polynomial, exponent times split ln 2.
// ---------------------------------------------------------------------------
DATA ·logConsts+0(SB)/4, $0x00800000 // minnorm
DATA ·logConsts+4(SB)/4, $0x00800000
DATA ·logConsts+8(SB)/4, $0x00800000
DATA ·logConsts+12(SB)/4, $0x00800000
DATA ·logConsts+16(SB)/4, $0x00800000
DATA ·logConsts+20(SB)/4, $0x00800000
DATA ·logConsts+24(SB)/4, $0x00800000
DATA ·logConsts+28(SB)/4, $0x00800000
DATA ·logConsts+32(SB)/4, $0x3F3504F3 // sqrthf
DATA ·logConsts+36(SB)/4, $0x3F3504F3
DATA ·logConsts+40(SB)/4, $0x3F3504F3
DATA ·logConsts+44(SB)/4, $0x3F3504F3
DATA ·logConsts+48(SB)/4, $0x3F3504F3
DATA ·logConsts+52(SB)/4, $0x3F3504F3
DATA ·logConsts+56(SB)/4, $0x3F3504F3
DATA ·logConsts+60(SB)/4, $0x3F3504F3
DATA ·logConsts+64(SB)/4, $0x3F800000 // one
DATA ·logConsts+68(SB)/4, $0x3F800000
DATA ·logConsts+72(SB)/4, $0x3F800000
DATA ·logConsts+76(SB)/4, $0x3F800000
DATA ·logConsts+80(SB)/4, $0x3F800000
DATA ·logConsts+84(SB)/4, $0x3F800000
DATA ·logConsts+88(SB)/4, $0x3F800000
DATA ·logConsts+92(SB)/4, $0x3F800000
DATA ·logConsts+96(SB)/4, $0x3F000000 // half
DATA ·logConsts+100(SB)/4, $0x3F000000
DATA ·logConsts+104(SB)/4, $0x3F000000
DATA ·logConsts+108(SB)/4, $0x3F000000
DATA ·logConsts+112(SB)/4, $0x3F000000
DATA ·logConsts+116(SB)/4, $0x3F000000
DATA ·logConsts+120(SB)/4, $0x3F000000
DATA ·logConsts+124(SB)/4, $0x3F000000
DATA ·logConsts+128(SB)/4, $0x3D9021BB // p0
DATA ·logConsts+132(SB)/4, $0x3D9021BB
DATA ·logConsts+136(SB)/4, $0x3D9021BB
DATA ·logConsts+140(SB)/4, $0x3D9021BB
DATA ·logConsts+144(SB)/4, $0x3D9021BB
DATA ·logConsts+148(SB)/4, $0x3D9021BB
DATA ·logConsts+152(SB)/4, $0x3D9021BB
DATA ·logConsts+156(SB)/4, $0x3D9021BB
DATA ·logConsts+160(SB)/4, $0xBDEBD1B8 // p1
DATA ·logConsts+164(SB)/4, $0xBDEBD1B8
DATA ·logConsts+168(SB)/4, $0xBDEBD1B8
DATA ·logConsts+172(SB)/4, $0xBDEBD1B8
DATA ·logConsts+176(SB)/4, $0xBDEBD1B8
DATA ·logConsts+180(SB)/4, $0xBDEBD1B8
DATA ·logConsts+184(SB)/4, $0xBDEBD1B8
DATA ·logConsts+188(SB)/4, $0xBDEBD1B8
DATA ·logConsts+192(SB)/4, $0x3DEF251A // p2
DATA ·logConsts+196(SB)/4, $0x3DEF251A
DATA ·logConsts+200(SB)/4, $0x3DEF251A
DATA ·logConsts+204(SB)/4, $0x3DEF251A
DATA ·logConsts+208(SB)/4, $0x3DEF251A
DATA ·logConsts+212(SB)/4, $0x3DEF251A
DATA ·logConsts+216(SB)/4, $0x3DEF251A
DATA ·logConsts+220(SB)/4, $0x3DEF251A
DATA ·logConsts+224(SB)/4, $0xBDFE5D4F // p3
DATA ·logConsts+228(SB)/4, $0xBDFE5D4F
DATA ·logConsts+232(SB)/4, $0xBDFE5D4F
DATA ·logConsts+236(SB)/4, $0xBDFE5D4F
DATA ·logConsts+240(SB)/4, $0xBDFE5D4F
DATA ·logConsts+244(SB)/4, $0xBDFE5D4F
DATA ·logConsts+248(SB)/4, $0xBDFE5D4F
DATA ·logConsts+252(SB)/4, $0xBDFE5D4F
DATA ·logConsts+256(SB)/4, $0x3E11E9BF // p4
DATA ·logConsts+260(SB)/4, $0x3E11E9BF
DATA ·logConsts+264(SB)/4, $0x3E11E9BF
DATA ·logConsts+268(SB)/4, $0x3E11E9BF
DATA ·logConsts+272(SB)/4, $0x3E11E9BF
DATA ·logConsts+276(SB)/4, $0x3E11E9BF
DATA ·logConsts+280(SB)/4, $0x3E11E9BF
DATA ·logConsts+284(SB)/4, $0x3E11E9BF
DATA ·logConsts+288(SB)/4, $0xBE2AAE50 // p5
DATA ·logConsts+292(SB)/4, $0xBE2AAE50
DATA ·logConsts+296(SB)/4, $0xBE2AAE50
DATA ·logConsts+300(SB)/4, $0xBE2AAE50
DATA ·logConsts+304(SB)/4, $0xBE2AAE50
DATA ·logConsts+308(SB)/4, $0xBE2AAE50
DATA ·logConsts+312(SB)/4, $0xBE2AAE50
DATA ·logConsts+316(SB)/4, $0xBE2AAE50
DATA ·logConsts+320(SB)/4, $0x3E4CCEAC // p6
DATA ·logConsts+324(SB)/4, $0x3E4CCEAC
DATA ·logConsts+328(SB)/4, $0x3E4CCEAC
DATA ·logConsts+332(SB)/4, $0x3E4CCEAC
DATA ·logConsts+336(SB)/4, $0x3E4CCEAC
DATA ·logConsts+340(SB)/4, $0x3E4CCEAC
DATA ·logConsts+344(SB)/4, $0x3E4CCEAC
DATA ·logConsts+348(SB)/4, $0x3E4CCEAC
DATA ·logConsts+352(SB)/4, $0xBE7FFFFC // p7
DATA ·logConsts+356(SB)/4, $0xBE7FFFFC
DATA ·logConsts+360(SB)/4, $0xBE7FFFFC
DATA ·logConsts+364(SB)/4, $0xBE7FFFFC
DATA ·logConsts+368(SB)/4, $0xBE7FFFFC
DATA ·logConsts+372(SB)/4, $0xBE7FFFFC
DATA ·logConsts+376(SB)/4, $0xBE7FFFFC
DATA ·logConsts+380(SB)/4, $0xBE7FFFFC
DATA ·logConsts+384(SB)/4, $0x3EAAAAAA // p8
DATA ·logConsts+388(SB)/4, $0x3EAAAAAA
DATA ·logConsts+392(SB)/4, $0x3EAAAAAA
DATA ·logConsts+396(SB)/4, $0x3EAAAAAA
DATA ·logConsts+400(SB)/4, $0x3EAAAAAA
DATA ·logConsts+404(SB)/4, $0x3EAAAAAA
DATA ·logConsts+408(SB)/4, $0x3EAAAAAA
DATA ·logConsts+412(SB)/4, $0x3EAAAAAA
DATA ·logConsts+416(SB)/4, $0xB95E8083 // ln2lo
DATA ·logConsts+420(SB)/4, $0xB95E8083
DATA ·logConsts+424(SB)/4, $0xB95E8083
DATA ·logConsts+428(SB)/4, $0xB95E8083
DATA ·logConsts+432(SB)/4, $0xB95E8083
DATA ·logConsts+436(SB)/4, $0xB95E8083
DATA ·logConsts+440(SB)/4, $0xB95E8083
DATA ·logConsts+444(SB)/4, $0xB95E8083
DATA ·logConsts+448(SB)/4, $0x3F318000 // ln2hi
DATA ·logConsts+452(SB)/4, $0x3F318000
DATA ·logConsts+456(SB)/4, $0x3F318000
DATA ·logConsts+460(SB)/4, $0x3F318000
DATA ·logConsts+464(SB)/4, $0x3F318000
DATA ·logConsts+468(SB)/4, $0x3F318000
DATA ·logConsts+472(SB)/4, $0x3F318000
DATA ·logConsts+476(SB)/4, $0x3F318000
DATA ·logConsts+480(SB)/4, $0x007FFFFF // mant
DATA ·logConsts+484(SB)/4, $0x007FFFFF
DATA ·logConsts+488(SB)/4, $0x007FFFFF
DATA ·logConsts+492(SB)/4, $0x007FFFFF
DATA ·logConsts+496(SB)/4, $0x007FFFFF
DATA ·logConsts+500(SB)/4, $0x007FFFFF
DATA ·logConsts+504(SB)/4, $0x007FFFFF
DATA ·logConsts+508(SB)/4, $0x007FFFFF
DATA ·logConsts+512(SB)/4, $0x0000007E // e126
DATA ·logConsts+516(SB)/4, $0x0000007E
DATA ·logConsts+520(SB)/4, $0x0000007E
DATA ·logConsts+524(SB)/4, $0x0000007E
DATA ·logConsts+528(SB)/4, $0x0000007E
DATA ·logConsts+532(SB)/4, $0x0000007E
DATA ·logConsts+536(SB)/4, $0x0000007E
DATA ·logConsts+540(SB)/4, $0x0000007E
DATA ·logConsts+544(SB)/4, $0x7F800000 // inf
DATA ·logConsts+548(SB)/4, $0x7F800000
DATA ·logConsts+552(SB)/4, $0x7F800000
DATA ·logConsts+556(SB)/4, $0x7F800000
DATA ·logConsts+560(SB)/4, $0x7F800000
DATA ·logConsts+564(SB)/4, $0x7F800000
DATA ·logConsts+568(SB)/4, $0x7F800000
DATA ·logConsts+572(SB)/4, $0x7F800000
DATA ·logConsts+576(SB)/4, $0xFF800000 // ninf
DATA ·logConsts+580(SB)/4, $0xFF800000
DATA ·logConsts+584(SB)/4, $0xFF800000
DATA ·logConsts+588(SB)/4, $0xFF800000
DATA ·logConsts+592(SB)/4, $0xFF800000
DATA ·logConsts+596(SB)/4, $0xFF800000
DATA ·logConsts+600(SB)/4, $0xFF800000
DATA ·logConsts+604(SB)/4, $0xFF800000
DATA ·logConsts+608(SB)/4, $0x7FC00000 // nan
DATA ·logConsts+612(SB)/4, $0x7FC00000
DATA ·logConsts+616(SB)/4, $0x7FC00000
DATA ·logConsts+620(SB)/4, $0x7FC00000
DATA ·logConsts+624(SB)/4, $0x7FC00000
DATA ·logConsts+628(SB)/4, $0x7FC00000
DATA ·logConsts+632(SB)/4, $0x7FC00000
DATA ·logConsts+636(SB)/4, $0x7FC00000
DATA ·logConsts+640(SB)/4, $0x00000000 // zero
DATA ·logConsts+644(SB)/4, $0x00000000
DATA ·logConsts+648(SB)/4, $0x00000000
DATA ·logConsts+652(SB)/4, $0x00000000
DATA ·logConsts+656(SB)/4, $0x00000000
DATA ·logConsts+660(SB)/4, $0x00000000
DATA ·logConsts+664(SB)/4, $0x00000000
DATA ·logConsts+668(SB)/4, $0x00000000
GLOBL ·logConsts(SB), RODATA|NOPTR, $672

TEXT ·logAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	SHRQ $3, CX
	JZ   logAVX2_done
	// Two vectors per iteration with independent registers: consecutive
	// iterations of the single-vector loop did not overlap (1.7 ns per
	// element on Skylake-SP, 1.4 on Apple M2 for the NEON version).
	MOVQ CX, BX
	SHRQ $1, BX
	JZ   logAVX2_tail
logAVX2_loop2:
	VMOVUPS (SI), Y0
	VMOVUPS 32(SI), Y8
	VMAXPS ·logConsts+0(SB), Y0, Y1
	VMAXPS ·logConsts+0(SB), Y8, Y9
	VPSRLD $23, Y1, Y2
	VPSRLD $23, Y9, Y10
	VPSUBD ·logConsts+512(SB), Y2, Y2
	VPSUBD ·logConsts+512(SB), Y10, Y10
	VCVTDQ2PS Y2, Y2
	VCVTDQ2PS Y10, Y10
	VPAND ·logConsts+480(SB), Y1, Y3
	VPAND ·logConsts+480(SB), Y9, Y11
	VPOR ·logConsts+96(SB), Y3, Y3
	VPOR ·logConsts+96(SB), Y11, Y11
	VCMPPS $1, ·logConsts+32(SB), Y3, Y4
	VCMPPS $1, ·logConsts+32(SB), Y11, Y12
	VANDPS Y4, Y3, Y5
	VANDPS Y12, Y11, Y13
	VANDPS ·logConsts+64(SB), Y4, Y4
	VANDPS ·logConsts+64(SB), Y12, Y12
	VSUBPS Y4, Y2, Y2
	VSUBPS Y12, Y10, Y10
	VSUBPS ·logConsts+64(SB), Y3, Y3
	VSUBPS ·logConsts+64(SB), Y11, Y11
	VADDPS Y5, Y3, Y3
	VADDPS Y13, Y11, Y11
	VMULPS Y3, Y3, Y7
	VMULPS Y11, Y11, Y15
	VMOVUPS ·logConsts+128(SB), Y6
	VMOVUPS ·logConsts+128(SB), Y14
	VFMADD213PS ·logConsts+160(SB), Y3, Y6
	VFMADD213PS ·logConsts+160(SB), Y11, Y14
	VFMADD213PS ·logConsts+192(SB), Y3, Y6
	VFMADD213PS ·logConsts+192(SB), Y11, Y14
	VFMADD213PS ·logConsts+224(SB), Y3, Y6
	VFMADD213PS ·logConsts+224(SB), Y11, Y14
	VFMADD213PS ·logConsts+256(SB), Y3, Y6
	VFMADD213PS ·logConsts+256(SB), Y11, Y14
	VFMADD213PS ·logConsts+288(SB), Y3, Y6
	VFMADD213PS ·logConsts+288(SB), Y11, Y14
	VFMADD213PS ·logConsts+320(SB), Y3, Y6
	VFMADD213PS ·logConsts+320(SB), Y11, Y14
	VFMADD213PS ·logConsts+352(SB), Y3, Y6
	VFMADD213PS ·logConsts+352(SB), Y11, Y14
	VFMADD213PS ·logConsts+384(SB), Y3, Y6
	VFMADD213PS ·logConsts+384(SB), Y11, Y14
	VMULPS Y3, Y6, Y6
	VMULPS Y11, Y14, Y14
	VMULPS Y7, Y6, Y6
	VMULPS Y15, Y14, Y14
	VFMADD231PS ·logConsts+416(SB), Y2, Y6
	VFMADD231PS ·logConsts+416(SB), Y10, Y14
	VFNMADD231PS ·logConsts+96(SB), Y7, Y6
	VFNMADD231PS ·logConsts+96(SB), Y15, Y14
	VADDPS Y3, Y6, Y6
	VADDPS Y11, Y14, Y14
	VFMADD231PS ·logConsts+448(SB), Y2, Y6
	VFMADD231PS ·logConsts+448(SB), Y10, Y14
	VCMPPS $0, ·logConsts+640(SB), Y0, Y4
	VCMPPS $0, ·logConsts+640(SB), Y8, Y12
	VBLENDVPS Y4, ·logConsts+576(SB), Y6, Y6
	VBLENDVPS Y12, ·logConsts+576(SB), Y14, Y14
	VCMPPS $1, ·logConsts+640(SB), Y0, Y4
	VCMPPS $1, ·logConsts+640(SB), Y8, Y12
	VBLENDVPS Y4, ·logConsts+608(SB), Y6, Y6
	VBLENDVPS Y12, ·logConsts+608(SB), Y14, Y14
	VCMPPS $0, ·logConsts+544(SB), Y0, Y4
	VCMPPS $0, ·logConsts+544(SB), Y8, Y12
	VBLENDVPS Y4, ·logConsts+544(SB), Y6, Y6
	VBLENDVPS Y12, ·logConsts+544(SB), Y14, Y14
	VMOVUPS Y6, (DX)
	VMOVUPS Y14, 32(DX)
	ADDQ $64, SI
	ADDQ $64, DX
	DECQ BX
	JNZ  logAVX2_loop2
	ANDQ $1, CX
	JZ   logAVX2_done
logAVX2_tail:
	VMOVUPS (SI), Y0
	VMAXPS ·logConsts+0(SB), Y0, Y1
	VPSRLD $23, Y1, Y2
	VPSUBD ·logConsts+512(SB), Y2, Y2
	VCVTDQ2PS Y2, Y2
	VPAND ·logConsts+480(SB), Y1, Y3
	VPOR ·logConsts+96(SB), Y3, Y3
	VCMPPS $1, ·logConsts+32(SB), Y3, Y4
	VANDPS Y4, Y3, Y5
	VANDPS ·logConsts+64(SB), Y4, Y4
	VSUBPS Y4, Y2, Y2
	VSUBPS ·logConsts+64(SB), Y3, Y3
	VADDPS Y5, Y3, Y3
	VMULPS Y3, Y3, Y7
	VMOVUPS ·logConsts+128(SB), Y6
	VFMADD213PS ·logConsts+160(SB), Y3, Y6
	VFMADD213PS ·logConsts+192(SB), Y3, Y6
	VFMADD213PS ·logConsts+224(SB), Y3, Y6
	VFMADD213PS ·logConsts+256(SB), Y3, Y6
	VFMADD213PS ·logConsts+288(SB), Y3, Y6
	VFMADD213PS ·logConsts+320(SB), Y3, Y6
	VFMADD213PS ·logConsts+352(SB), Y3, Y6
	VFMADD213PS ·logConsts+384(SB), Y3, Y6
	VMULPS Y3, Y6, Y6
	VMULPS Y7, Y6, Y6
	VFMADD231PS ·logConsts+416(SB), Y2, Y6
	VFNMADD231PS ·logConsts+96(SB), Y7, Y6
	VADDPS Y3, Y6, Y6
	VFMADD231PS ·logConsts+448(SB), Y2, Y6
	VCMPPS $0, ·logConsts+640(SB), Y0, Y4
	VBLENDVPS Y4, ·logConsts+576(SB), Y6, Y6
	VCMPPS $1, ·logConsts+640(SB), Y0, Y4
	VBLENDVPS Y4, ·logConsts+608(SB), Y6, Y6
	VCMPPS $0, ·logConsts+544(SB), Y0, Y4
	VBLENDVPS Y4, ·logConsts+544(SB), Y6, Y6
	VMOVUPS Y6, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	DECQ CX
	JNZ  logAVX2_tail
logAVX2_done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func sqrtAVX2(x, z *float32, n int)     (n % 8 == 0)
// ---------------------------------------------------------------------------
TEXT ·sqrtAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ z+8(FP), DX
	MOVQ n+16(FP), CX
	SHRQ $3, CX
	JZ   sqrt_done
sqrt_loop:
	VSQRTPS (SI), Y0
	VMOVUPS Y0, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	DECQ CX
	JNZ  sqrt_loop
sqrt_done:
	VZEROUPPER
	RET

// func dotNormsAVX2(x, y *float32, n int, out *[3]float32)
//
// One pass: out = {x·y, x·x, y·y}, two accumulators per sum.
TEXT ·dotNormsAVX2(SB), NOSPLIT, $0-32
	MOVQ x+0(FP), SI
	MOVQ y+8(FP), DI
	MOVQ n+16(FP), CX
	MOVQ out+24(FP), DX
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	MOVQ CX, BX
	SHRQ $4, BX
	JZ   dn_tail8
dn_loop16:
	VMOVUPS (SI), Y6
	VMOVUPS 32(SI), Y7
	VMOVUPS (DI), Y8
	VMOVUPS 32(DI), Y9
	VFMADD231PS Y8, Y6, Y0
	VFMADD231PS Y9, Y7, Y1
	VFMADD231PS Y6, Y6, Y2
	VFMADD231PS Y7, Y7, Y3
	VFMADD231PS Y8, Y8, Y4
	VFMADD231PS Y9, Y9, Y5
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ BX
	JNZ  dn_loop16
dn_tail8:
	MOVQ CX, BX
	ANDQ $15, BX
	SHRQ $3, BX
	JZ   dn_reduce
	VMOVUPS (SI), Y6
	VMOVUPS (DI), Y8
	VFMADD231PS Y8, Y6, Y0
	VFMADD231PS Y6, Y6, Y2
	VFMADD231PS Y8, Y8, Y4
	ADDQ $32, SI
	ADDQ $32, DI
dn_reduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y5, Y4, Y4
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VEXTRACTF128 $1, Y2, X3
	VADDPS X3, X2, X2
	VEXTRACTF128 $1, Y4, X5
	VADDPS X5, X4, X4
	ANDQ $7, CX
	JZ   dn_hsum
dn_loop1:
	VMOVSS (SI), X6
	VMOVSS (DI), X8
	VFMADD231SS X8, X6, X0
	VFMADD231SS X6, X6, X2
	VFMADD231SS X8, X8, X4
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  dn_loop1
dn_hsum:
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X2, X2, X2
	VHADDPS X2, X2, X2
	VHADDPS X4, X4, X4
	VHADDPS X4, X4, X4
	VMOVSS X0, (DX)
	VMOVSS X2, 4(DX)
	VMOVSS X4, 8(DX)
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// Row-major-A variant of the 6×16 kernel (T-046): A broadcasts from a
// row-major tile with row stride lda floats (R11 = lda, R12 = 3·lda,
// R13 = 5·lda bytes), B a packed 16-wide panel, C as in gemmAVX2.
//
// func gemmRMAVX2(k int, a *float32, lda int, b, c *float32, ldc int)
// ---------------------------------------------------------------------------
#define KSTEPRM6(AOFF, BOFF) \
	PREFETCHT0 (BOFF+512)(BX) \
	VMOVUPS BOFF(BX), Y12 \
	VMOVUPS BOFF+32(BX), Y13 \
	VBROADCASTSS AOFF(AX), Y14 \
	VBROADCASTSS AOFF(AX)(R11*1), Y15 \
	VFMADD231PS Y12, Y14, Y0 \
	VFMADD231PS Y13, Y14, Y1 \
	VFMADD231PS Y12, Y15, Y2 \
	VFMADD231PS Y13, Y15, Y3 \
	VBROADCASTSS AOFF(AX)(R11*2), Y14 \
	VBROADCASTSS AOFF(AX)(R12*1), Y15 \
	VFMADD231PS Y12, Y14, Y4 \
	VFMADD231PS Y13, Y14, Y5 \
	VFMADD231PS Y12, Y15, Y6 \
	VFMADD231PS Y13, Y15, Y7 \
	VBROADCASTSS AOFF(AX)(R11*4), Y14 \
	VBROADCASTSS AOFF(AX)(R13*1), Y15 \
	VFMADD231PS Y12, Y14, Y8 \
	VFMADD231PS Y13, Y14, Y9 \
	VFMADD231PS Y12, Y15, Y10 \
	VFMADD231PS Y13, Y15, Y11

TEXT ·gemmRMAVX2(SB), NOSPLIT, $0-48
	MOVQ $1, R10
	JMP  ·gemmRMAVX2Body(SB)

TEXT ·gemmRMZeroAVX2(SB), NOSPLIT, $0-48
	XORQ R10, R10
	JMP  ·gemmRMAVX2Body(SB)

TEXT ·gemmRMAVX2Body(SB), NOSPLIT, $0-48
	MOVQ k+0(FP), CX
	MOVQ a+8(FP), AX
	MOVQ lda+16(FP), R11
	MOVQ b+24(FP), BX
	MOVQ c+32(FP), DX
	MOVQ ldc+40(FP), R8
	SHLQ $2, R8
	SHLQ $2, R11
	LEAQ (R11)(R11*2), R12
	LEAQ (R11)(R11*4), R13
	TESTQ CX, CX
	JZ   donerm6
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
	JZ   ktailrm6
klooprm6:
	KSTEPRM6(0, 0)
	KSTEPRM6(4, 64)
	KSTEPRM6(8, 128)
	KSTEPRM6(12, 192)
	ADDQ $16, AX
	ADDQ $256, BX
	DECQ R9
	JNZ  klooprm6
ktailrm6:
	ANDQ $3, CX
	JZ   storerm6
klooprm6t:
	KSTEPRM6(0, 0)
	ADDQ $4, AX
	ADDQ $64, BX
	DECQ CX
	JNZ  klooprm6t
storerm6:
	TESTQ R10, R10
	JZ    storezrm6
	CROW(Y0, Y1)
	CROW(Y2, Y3)
	CROW(Y4, Y5)
	CROW(Y6, Y7)
	CROW(Y8, Y9)
	CROW(Y10, Y11)
	VZEROUPPER
	RET
storezrm6:
	CROWZ(Y0, Y1)
	CROWZ(Y2, Y3)
	CROWZ(Y4, Y5)
	CROWZ(Y6, Y7)
	CROWZ(Y8, Y9)
	CROWZ(Y10, Y11)
donerm6:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// A and B both row-major (T-050), 6×16: the small-product kernel.
//
// func gemmRMBAVX2(k int, a *float32, lda int, b *float32, ldb int, c *float32, ldc int)
// ---------------------------------------------------------------------------
#define KSTEPRMB6(AOFF) \
	PREFETCHT0 (BX)(R14*4) \
	VMOVUPS (BX), Y12 \
	VMOVUPS 32(BX), Y13 \
	VBROADCASTSS AOFF(AX), Y14 \
	VBROADCASTSS AOFF(AX)(R11*1), Y15 \
	VFMADD231PS Y12, Y14, Y0 \
	VFMADD231PS Y13, Y14, Y1 \
	VFMADD231PS Y12, Y15, Y2 \
	VFMADD231PS Y13, Y15, Y3 \
	VBROADCASTSS AOFF(AX)(R11*2), Y14 \
	VBROADCASTSS AOFF(AX)(R12*1), Y15 \
	VFMADD231PS Y12, Y14, Y4 \
	VFMADD231PS Y13, Y14, Y5 \
	VFMADD231PS Y12, Y15, Y6 \
	VFMADD231PS Y13, Y15, Y7 \
	VBROADCASTSS AOFF(AX)(R11*4), Y14 \
	VBROADCASTSS AOFF(AX)(R13*1), Y15 \
	VFMADD231PS Y12, Y14, Y8 \
	VFMADD231PS Y13, Y14, Y9 \
	VFMADD231PS Y12, Y15, Y10 \
	VFMADD231PS Y13, Y15, Y11 \
	ADDQ R14, BX

TEXT ·gemmRMBAVX2(SB), NOSPLIT, $0-56
	MOVQ $1, R10
	JMP  ·gemmRMBAVX2Body(SB)

TEXT ·gemmRMBZeroAVX2(SB), NOSPLIT, $0-56
	XORQ R10, R10
	JMP  ·gemmRMBAVX2Body(SB)

TEXT ·gemmRMBAVX2Body(SB), NOSPLIT, $0-56
	MOVQ k+0(FP), CX
	MOVQ a+8(FP), AX
	MOVQ lda+16(FP), R11
	MOVQ b+24(FP), BX
	MOVQ ldb+32(FP), R14
	MOVQ c+40(FP), DX
	MOVQ ldc+48(FP), R8
	SHLQ $2, R8
	SHLQ $2, R11
	SHLQ $2, R14
	LEAQ (R11)(R11*2), R12
	LEAQ (R11)(R11*4), R13
	TESTQ CX, CX
	JZ   donermb6
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
	JZ   ktailrmb6
klooprmb6:
	KSTEPRMB6(0)
	KSTEPRMB6(4)
	KSTEPRMB6(8)
	KSTEPRMB6(12)
	ADDQ $16, AX
	DECQ R9
	JNZ  klooprmb6
ktailrmb6:
	ANDQ $3, CX
	JZ   storermb6
klooprmb6t:
	KSTEPRMB6(0)
	ADDQ $4, AX
	DECQ CX
	JNZ  klooprmb6t
storermb6:
	TESTQ R10, R10
	JZ    storezrmb6
	CROW(Y0, Y1)
	CROW(Y2, Y3)
	CROW(Y4, Y5)
	CROW(Y6, Y7)
	CROW(Y8, Y9)
	CROW(Y10, Y11)
	VZEROUPPER
	RET
storezrmb6:
	CROWZ(Y0, Y1)
	CROWZ(Y2, Y3)
	CROWZ(Y4, Y5)
	CROWZ(Y6, Y7)
	CROWZ(Y8, Y9)
	CROWZ(Y10, Y11)
donermb6:
	VZEROUPPER
	RET
