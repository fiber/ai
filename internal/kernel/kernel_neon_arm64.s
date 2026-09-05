//go:build arm64

#include "textflag.h"

// float32 NEON kernels.
//
// The Go assembler lacks mnemonics for the vector floating-point arithmetic
// instructions, so they are emitted as raw encodings (all in the .4S
// arrangement). Operand order of the macros follows the ARM manual:
// OP(Vd, Vn, Vm) computes Vd = Vn op Vm. Registers are given as numbers.

// FADD Vd.4S, Vn.4S, Vm.4S
#define VFADD4(Vd, Vn, Vm) WORD $(0x4E20D400 | (Vm<<16) | (Vn<<5) | Vd)
// FSUB Vd.4S, Vn.4S, Vm.4S
#define VFSUB4(Vd, Vn, Vm) WORD $(0x4EA0D400 | (Vm<<16) | (Vn<<5) | Vd)
// FMUL Vd.4S, Vn.4S, Vm.4S
#define VFMUL4(Vd, Vn, Vm) WORD $(0x6E20DC00 | (Vm<<16) | (Vn<<5) | Vd)
// FDIV Vd.4S, Vn.4S, Vm.4S
#define VFDIV4(Vd, Vn, Vm) WORD $(0x6E20FC00 | (Vm<<16) | (Vn<<5) | Vd)
// FMAX Vd.4S, Vn.4S, Vm.4S
#define VFMAX4(Vd, Vn, Vm) WORD $(0x4E20F400 | (Vm<<16) | (Vn<<5) | Vd)
// FADDP Vd.4S, Vn.4S, Vm.4S (pairwise add)
#define VFADDP4(Vd, Vn, Vm) WORD $(0x6E20D400 | (Vm<<16) | (Vn<<5) | Vd)
// FMAXV Sd, Vn.4S (horizontal max)
#define VFMAXV4(Sd, Vn) WORD $(0x6E30F800 | (Vn<<5) | Sd)
// FMLA Vd.4S, Vn.4S, Vm.S[idx]  (Vd += Vn * Vm[idx])
#define FMLA_E0(Vd, Vn, Vm) WORD $(0x4F801000 | (Vm<<16) | (Vn<<5) | Vd)
#define FMLA_E1(Vd, Vn, Vm) WORD $(0x4FA01000 | (Vm<<16) | (Vn<<5) | Vd)
#define FMLA_E2(Vd, Vn, Vm) WORD $(0x4F801800 | (Vm<<16) | (Vn<<5) | Vd)
#define FMLA_E3(Vd, Vn, Vm) WORD $(0x4FA01800 | (Vm<<16) | (Vn<<5) | Vd)

// All multi-line macros are defined here, before the first TEXT, because
// go vet's asmdecl attributes macro body lines to the preceding function.

// Binary element-wise kernels: z = x OP y
// func xxxNEON(x, y, z *float32, n int)
#define BINARY_BODY(VOP, SOP) \
	MOVD x+0(FP), R0 \
	MOVD y+8(FP), R1 \
	MOVD z+16(FP), R2 \
	MOVD n+24(FP), R3 \
	LSR  $4, R3, R4 \
	CBZ  R4, tail4 \
loop16: \
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4] \
	VLD1.P 64(R1), [V4.S4, V5.S4, V6.S4, V7.S4] \
	VOP(0, 0, 4) \
	VOP(1, 1, 5) \
	VOP(2, 2, 6) \
	VOP(3, 3, 7) \
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2) \
	SUBS $1, R4, R4 \
	BNE  loop16 \
tail4: \
	AND  $15, R3, R4 \
	LSR  $2, R4, R4 \
	CBZ  R4, tail1 \
loop4: \
	VLD1.P 16(R0), [V0.S4] \
	VLD1.P 16(R1), [V4.S4] \
	VOP(0, 0, 4) \
	VST1.P [V0.S4], 16(R2) \
	SUBS $1, R4, R4 \
	BNE  loop4 \
tail1: \
	AND  $3, R3, R3 \
	CBZ  R3, done \
loop1: \
	FMOVS (R0), F0 \
	FMOVS (R1), F4 \
	SOP   F4, F0, F0 \
	FMOVS F0, (R2) \
	ADD   $4, R0, R0 \
	ADD   $4, R1, R1 \
	ADD   $4, R2, R2 \
	SUBS  $1, R3, R3 \
	BNE   loop1 \
done: \
	RET

// Scalar-broadcast kernels: z = x OP s
// func xxxNEON(x, z *float32, s float32, n int)
#define SCALAR_BODY(VOP, SOP) \
	MOVD  x+0(FP), R0 \
	MOVD  z+8(FP), R2 \
	FMOVS s+16(FP), F4 \
	MOVD  n+24(FP), R3 \
	VDUP  V4.S[0], V4.S4 \
	LSR   $4, R3, R4 \
	CBZ   R4, tail4 \
loop16: \
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4] \
	VOP(0, 0, 4) \
	VOP(1, 1, 4) \
	VOP(2, 2, 4) \
	VOP(3, 3, 4) \
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2) \
	SUBS $1, R4, R4 \
	BNE  loop16 \
tail4: \
	AND  $15, R3, R4 \
	LSR  $2, R4, R4 \
	CBZ  R4, tail1 \
loop4: \
	VLD1.P 16(R0), [V0.S4] \
	VOP(0, 0, 4) \
	VST1.P [V0.S4], 16(R2) \
	SUBS $1, R4, R4 \
	BNE  loop4 \
tail1: \
	AND  $3, R3, R3 \
	CBZ  R3, done \
loop1: \
	FMOVS (R0), F0 \
	SOP   F4, F0, F0 \
	FMOVS F0, (R2) \
	ADD   $4, R0, R0 \
	ADD   $4, R2, R2 \
	SUBS  $1, R3, R3 \
	BNE   loop1 \
done: \
	RET

// GEMM micro-kernel k-step (MR=8, NR=12): A in V24/V25, B in V26..V28,
// accumulators V0..V23 (row r -> V(3r), V(3r+1), V(3r+2)).
#define KSTEP \
	VLD1.P 32(R0), [V24.S4, V25.S4] \
	VLD1.P 48(R1), [V26.S4, V27.S4, V28.S4] \
	FMLA_E0(0, 26, 24) \
	FMLA_E0(1, 27, 24) \
	FMLA_E0(2, 28, 24) \
	FMLA_E1(3, 26, 24) \
	FMLA_E1(4, 27, 24) \
	FMLA_E1(5, 28, 24) \
	FMLA_E2(6, 26, 24) \
	FMLA_E2(7, 27, 24) \
	FMLA_E2(8, 28, 24) \
	FMLA_E3(9, 26, 24) \
	FMLA_E3(10, 27, 24) \
	FMLA_E3(11, 28, 24) \
	FMLA_E0(12, 26, 25) \
	FMLA_E0(13, 27, 25) \
	FMLA_E0(14, 28, 25) \
	FMLA_E1(15, 26, 25) \
	FMLA_E1(16, 27, 25) \
	FMLA_E1(17, 28, 25) \
	FMLA_E2(18, 26, 25) \
	FMLA_E2(19, 27, 25) \
	FMLA_E2(20, 28, 25) \
	FMLA_E3(21, 26, 25) \
	FMLA_E3(22, 27, 25) \
	FMLA_E3(23, 28, 25)

// CROW(a, b, c, Va, Vb, Vc): C row += accumulators a,b,c; advance C by ldc.
#define CROW(A, B, C, VA, VB, VC) \
	VLD1 (R2), [V24.S4, V25.S4, V26.S4] \
	VFADD4(A, A, 24) \
	VFADD4(B, B, 25) \
	VFADD4(C, C, 26) \
	VST1 [VA.S4, VB.S4, VC.S4], (R2) \
	ADD  R4, R2, R2

// ---------------------------------------------------------------------------
// Binary element-wise kernels: z = x OP y
// func xxxNEON(x, y, z *float32, n int)
// ---------------------------------------------------------------------------


TEXT ·addNEON(SB), NOSPLIT, $0-32
	BINARY_BODY(VFADD4, FADDS)

TEXT ·subNEON(SB), NOSPLIT, $0-32
	BINARY_BODY(VFSUB4, FSUBS)

TEXT ·mulNEON(SB), NOSPLIT, $0-32
	BINARY_BODY(VFMUL4, FMULS)

TEXT ·divNEON(SB), NOSPLIT, $0-32
	BINARY_BODY(VFDIV4, FDIVS)

TEXT ·maximumNEON(SB), NOSPLIT, $0-32
	BINARY_BODY(VFMAX4, FMAXS)

// ---------------------------------------------------------------------------
// Scalar-broadcast kernels: z = x OP s
// func xxxNEON(x, z *float32, s float32, n int)
// ---------------------------------------------------------------------------


TEXT ·addScalarNEON(SB), NOSPLIT, $0-32
	SCALAR_BODY(VFADD4, FADDS)

TEXT ·scaleNEON(SB), NOSPLIT, $0-32
	SCALAR_BODY(VFMUL4, FMULS)

TEXT ·maxScalarNEON(SB), NOSPLIT, $0-32
	SCALAR_BODY(VFMAX4, FMAXS)

// ---------------------------------------------------------------------------
// func axpyNEON(x, y *float32, alpha float32, n int)   y += alpha * x
// ---------------------------------------------------------------------------
TEXT ·axpyNEON(SB), NOSPLIT, $0-32
	MOVD  x+0(FP), R0
	MOVD  y+8(FP), R1
	FMOVS alpha+16(FP), F8
	MOVD  n+24(FP), R3
	VDUP  V8.S[0], V8.S4
	LSR   $4, R3, R4
	CBZ   R4, tail4
loop16:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1   (R1), [V4.S4, V5.S4, V6.S4, V7.S4]
	VFMLA  V8.S4, V0.S4, V4.S4
	VFMLA  V8.S4, V1.S4, V5.S4
	VFMLA  V8.S4, V2.S4, V6.S4
	VFMLA  V8.S4, V3.S4, V7.S4
	VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R1)
	SUBS   $1, R4, R4
	BNE    loop16
tail4:
	AND  $15, R3, R4
	LSR  $2, R4, R4
	CBZ  R4, tail1
loop4:
	VLD1.P 16(R0), [V0.S4]
	VLD1   (R1), [V4.S4]
	VFMLA  V8.S4, V0.S4, V4.S4
	VST1.P [V4.S4], 16(R1)
	SUBS   $1, R4, R4
	BNE    loop4
tail1:
	AND  $3, R3, R3
	CBZ  R3, done
loop1:
	FMOVS (R0), F0
	FMOVS (R1), F4
	FMULS F8, F0, F0
	FADDS F0, F4, F4
	FMOVS F4, (R1)
	ADD   $4, R0, R0
	ADD   $4, R1, R1
	SUBS  $1, R3, R3
	BNE   loop1
done:
	RET

// ---------------------------------------------------------------------------
// func dotNEON(x, y *float32, n int) float32
// ---------------------------------------------------------------------------
TEXT ·dotNEON(SB), NOSPLIT, $0-28
	MOVD x+0(FP), R0
	MOVD y+8(FP), R1
	MOVD n+16(FP), R3
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	LSR  $4, R3, R4
	CBZ  R4, tail4
loop16:
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	VFMLA  V9.S4, V5.S4, V1.S4
	VFMLA  V10.S4, V6.S4, V2.S4
	VFMLA  V11.S4, V7.S4, V3.S4
	SUBS   $1, R4, R4
	BNE    loop16
tail4:
	AND  $15, R3, R4
	LSR  $2, R4, R4
	CBZ  R4, reduce
loop4:
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R1), [V8.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	SUBS   $1, R4, R4
	BNE    loop4
reduce:
	VFADD4(0, 0, 1)
	VFADD4(2, 2, 3)
	VFADD4(0, 0, 2)
	VFADDP4(0, 0, 0)
	VFADDP4(0, 0, 0)
	AND  $3, R3, R3
	CBZ  R3, done
loop1:
	FMOVS (R0), F4
	FMOVS (R1), F8
	FMULS F8, F4, F4
	FADDS F4, F0, F0
	ADD   $4, R0, R0
	ADD   $4, R1, R1
	SUBS  $1, R3, R3
	BNE   loop1
done:
	FMOVS F0, ret+24(FP)
	RET

// ---------------------------------------------------------------------------
// func sumNEON(x *float32, n int) float32
// ---------------------------------------------------------------------------
TEXT ·sumNEON(SB), NOSPLIT, $0-20
	MOVD x+0(FP), R0
	MOVD n+8(FP), R3
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	LSR  $4, R3, R4
	CBZ  R4, tail4
loop16:
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VFADD4(0, 0, 4)
	VFADD4(1, 1, 5)
	VFADD4(2, 2, 6)
	VFADD4(3, 3, 7)
	SUBS $1, R4, R4
	BNE  loop16
tail4:
	AND  $15, R3, R4
	LSR  $2, R4, R4
	CBZ  R4, reduce
loop4:
	VLD1.P 16(R0), [V4.S4]
	VFADD4(0, 0, 4)
	SUBS $1, R4, R4
	BNE  loop4
reduce:
	VFADD4(0, 0, 1)
	VFADD4(2, 2, 3)
	VFADD4(0, 0, 2)
	VFADDP4(0, 0, 0)
	VFADDP4(0, 0, 0)
	AND  $3, R3, R3
	CBZ  R3, done
loop1:
	FMOVS (R0), F4
	FADDS F4, F0, F0
	ADD   $4, R0, R0
	SUBS  $1, R3, R3
	BNE   loop1
done:
	FMOVS F0, ret+16(FP)
	RET

// ---------------------------------------------------------------------------
// func maxNEON(x *float32, n int) float32     (n >= 1)
// ---------------------------------------------------------------------------
TEXT ·maxNEON(SB), NOSPLIT, $0-20
	MOVD  x+0(FP), R0
	MOVD  n+8(FP), R3
	VLD1R (R0), [V0.S4]
	VMOV  V0.B16, V1.B16
	VMOV  V0.B16, V2.B16
	VMOV  V0.B16, V3.B16
	LSR   $4, R3, R4
	CBZ   R4, tail4
loop16:
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VFMAX4(0, 0, 4)
	VFMAX4(1, 1, 5)
	VFMAX4(2, 2, 6)
	VFMAX4(3, 3, 7)
	SUBS $1, R4, R4
	BNE  loop16
tail4:
	AND  $15, R3, R4
	LSR  $2, R4, R4
	CBZ  R4, reduce
loop4:
	VLD1.P 16(R0), [V4.S4]
	VFMAX4(0, 0, 4)
	SUBS $1, R4, R4
	BNE  loop4
reduce:
	VFMAX4(0, 0, 1)
	VFMAX4(2, 2, 3)
	VFMAX4(0, 0, 2)
	VFMAXV4(0, 0)
	AND  $3, R3, R3
	CBZ  R3, done
loop1:
	FMOVS (R0), F4
	FMAXS F4, F0, F0
	ADD   $4, R0, R0
	SUBS  $1, R3, R3
	BNE   loop1
done:
	FMOVS F0, ret+16(FP)
	RET

// ---------------------------------------------------------------------------
// GEMM micro-kernel, MR=8, NR=12:  C[8×12] += A[8×k] · B[k×12]
// func gemmNEON(k int, a, b, c *float32, ldc int)
//
// A is packed k-major (8 floats per k), B is packed k-major (12 floats per k).
// Accumulators: row r of C lives in V(3r), V(3r+1), V(3r+2) -> V0..V23.
// V24/V25 hold the 8 A values of the current k, V26..V28 the 12 B values.
// ---------------------------------------------------------------------------


// CROW(a, b, c, Va, Vb, Vc): C row += accumulators a,b,c; advance C by ldc.

TEXT ·gemmNEON(SB), NOSPLIT, $0-40
	MOVD k+0(FP), R3
	MOVD a+8(FP), R0
	MOVD b+16(FP), R1
	MOVD c+24(FP), R2
	MOVD ldc+32(FP), R4
	LSL  $2, R4, R4
	CBZ  R3, done
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16
	LSR  $2, R3, R5
	CBZ  R5, ktail
kloop4:
	KSTEP
	KSTEP
	KSTEP
	KSTEP
	SUBS $1, R5, R5
	BNE  kloop4
ktail:
	AND  $3, R3, R3
	CBZ  R3, store
kloop1:
	KSTEP
	SUBS $1, R3, R3
	BNE  kloop1
store:
	CROW(0, 1, 2, V0, V1, V2)
	CROW(3, 4, 5, V3, V4, V5)
	CROW(6, 7, 8, V6, V7, V8)
	CROW(9, 10, 11, V9, V10, V11)
	CROW(12, 13, 14, V12, V13, V14)
	CROW(15, 16, 17, V15, V16, V17)
	CROW(18, 19, 20, V18, V19, V20)
	CROW(21, 22, 23, V21, V22, V23)
done:
	RET

// ---------------------------------------------------------------------------
// func expNEON(x, z *float32, n int)      (n % 4 == 0)
//
// Vectorised exp, 4 floats per iteration; see genericExp for the algorithm.
// Constants live in V16..V28, the working set in V0..V4.
// ---------------------------------------------------------------------------
#define VFMIN4(Vd, Vn, Vm) WORD $(0x4EA0F400 | (Vm<<16) | (Vn<<5) | Vd)

TEXT ·expNEON(SB), NOSPLIT, $0-24
	MOVD x+0(FP), R0
	MOVD z+8(FP), R2
	MOVD n+16(FP), R3
	MOVD $0x3FB8AA3B, R4  // log2e
	VDUP R4, V16.S4
	MOVD $0x4B400000, R4  // magic 1.5·2^23
	VDUP R4, V17.S4
	MOVD $0x3F318000, R4  // ln2 hi
	VDUP R4, V18.S4
	MOVD $0xB95E8083, R4  // ln2 lo
	VDUP R4, V19.S4
	MOVD $0x42B0C0A5, R4  // clamp hi
	VDUP R4, V20.S4
	MOVD $0xC2AE0000, R4  // clamp lo (-87.0)
	VDUP R4, V21.S4
	MOVD $0x39506967, R4  // c0
	VDUP R4, V22.S4
	MOVD $0x3AB743CE, R4  // c1
	VDUP R4, V23.S4
	MOVD $0x3C088908, R4  // c2
	VDUP R4, V24.S4
	MOVD $0x3D2AA9C1, R4  // c3
	VDUP R4, V25.S4
	MOVD $0x3E2AAAAA, R4  // c4
	VDUP R4, V26.S4
	MOVD $0x3F000000, R4  // c5
	VDUP R4, V27.S4
	MOVD $0x3F800000, R4  // 1.0
	VDUP R4, V28.S4
	LSR  $2, R3, R3
	CBZ  R3, done
loop:
	VLD1.P 16(R0), [V0.S4]
	VFMAX4(0, 0, 21)               // x = max(x, lo)
	VFMIN4(0, 0, 20)               // x = min(x, hi)
	VFMUL4(1, 0, 16)               // t = x·log2e
	VFADD4(1, 1, 17)               // t += magic  (low mantissa bits now hold n)
	VFSUB4(2, 1, 17)               // nf = t - magic
	VFMLS V18.S4, V2.S4, V0.S4     // r = x - nf·ln2hi
	VFMLS V19.S4, V2.S4, V0.S4     // r -= nf·ln2lo
	VMOV  V22.B16, V3.B16          // p = c0
	VMOV  V23.B16, V4.B16
	VFMLA V0.S4, V3.S4, V4.S4      // p = c1 + p·r
	VMOV  V24.B16, V3.B16
	VFMLA V0.S4, V4.S4, V3.S4      // p = c2 + p·r
	VMOV  V25.B16, V4.B16
	VFMLA V0.S4, V3.S4, V4.S4      // p = c3 + p·r
	VMOV  V26.B16, V3.B16
	VFMLA V0.S4, V4.S4, V3.S4      // p = c4 + p·r
	VMOV  V27.B16, V4.B16
	VFMLA V0.S4, V3.S4, V4.S4      // p = c5 + p·r
	VFMUL4(3, 0, 0)                // r²
	VFMUL4(4, 4, 3)                // p·r²
	VFADD4(4, 4, 0)                // + r
	VFADD4(4, 4, 28)               // + 1
	VSUB  V17.S4, V1.S4, V1.S4     // n = bits(t) - bits(magic)
	VSHL  $23, V1.S4, V1.S4        // n << 23
	VADD  V1.S4, V4.S4, V4.S4      // p · 2^n via the exponent field
	VST1.P [V4.S4], 16(R2)
	SUBS $1, R3, R3
	BNE  loop
done:
	RET
