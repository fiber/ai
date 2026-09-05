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

// ---------------------------------------------------------------------------
// func tanhNEON(x, z *float32, n int)     (n % 4 == 0)
//
// Rational approximation, see genericTanh. Constants in V16..V29.
// ---------------------------------------------------------------------------
#define VFABS4(Vd, Vn) WORD $(0x4EA0F800 | (Vn<<5) | Vd)
#define VFCMGT4(Vd, Vn, Vm) WORD $(0x6EA0E400 | (Vm<<16) | (Vn<<5) | Vd)
#define VFCMEQ4(Vd, Vn, Vm) WORD $(0x4E20E400 | (Vm<<16) | (Vn<<5) | Vd)
#define VBSL16(Vd, Vn, Vm) WORD $(0x6E601C00 | (Vm<<16) | (Vn<<5) | Vd)
#define VSCVTF4(Vd, Vn) WORD $(0x4E21D800 | (Vn<<5) | Vd)
#define VFCMGE4(Vd, Vn, Vm) WORD $(0x6E20E400 | (Vm<<16) | (Vn<<5) | Vd)

TEXT ·tanhNEON(SB), NOSPLIT, $0-24
	MOVD x+0(FP), R0
	MOVD z+8(FP), R2
	MOVD n+16(FP), R3
	MOVD $0x40FFF644, R4 // clamp
	VDUP R4, V16.S4
	MOVD $0xC0FFF644, R4 // nclamp
	VDUP R4, V17.S4
	MOVD $0x39D1B717, R4 // tiny
	VDUP R4, V18.S4
	MOVD $0x3BA059DC, R4 // a1
	VDUP R4, V19.S4
	MOVD $0x3A270DED, R4 // a3
	VDUP R4, V20.S4
	MOVD $0x3779434A, R4 // a5
	VDUP R4, V21.S4
	MOVD $0x335C0041, R4 // a7
	VDUP R4, V22.S4
	MOVD $0xAEBD37FF, R4 // a9
	VDUP R4, V23.S4
	MOVD $0x2A61337E, R4 // a11
	VDUP R4, V24.S4
	MOVD $0xA59F25C0, R4 // a13
	VDUP R4, V25.S4
	MOVD $0x3BA059DD, R4 // b0
	VDUP R4, V26.S4
	MOVD $0x3B14AA05, R4 // b2
	VDUP R4, V27.S4
	MOVD $0x38F895D6, R4 // b4
	VDUP R4, V28.S4
	MOVD $0x35A0D3D8, R4 // b6
	VDUP R4, V29.S4
	MOVD $0x3F800000, R4 // 1.0
	VDUP R4, V8.S4
	MOVD $0x80000000, R4 // sign mask
	VDUP R4, V9.S4
	MOVD $0x41100000, R4 // 9.0: tanh is exactly ±1 in float32 beyond
	VDUP R4, V10.S4
	LSR  $2, R3, R3
	CBZ  R3, tdone
tloop:
	VLD1.P 16(R0), [V0.S4]
	VFABS4(6, 0)                       // |x|
	VAND  V9.B16, V0.B16, V11.B16      // sign bit
	VORR  V8.B16, V11.B16, V11.B16     // ±1
	VFMAX4(0, 0, 17)                  // clamp
	VFMIN4(0, 0, 16)
	VFMUL4(1, 0, 0)                    // x²
	VMOV  V24.B16, V3.B16              // p = a11
	VFMLA V1.S4, V25.S4, V3.S4          // p += a13·x²
	VMOV  V23.B16, V2.B16
	VFMLA V1.S4, V3.S4, V2.S4          // p = a9 + p·x²
	VMOV  V22.B16, V3.B16
	VFMLA V1.S4, V2.S4, V3.S4
	VMOV  V21.B16, V2.B16
	VFMLA V1.S4, V3.S4, V2.S4
	VMOV  V20.B16, V3.B16
	VFMLA V1.S4, V2.S4, V3.S4
	VMOV  V19.B16, V2.B16
	VFMLA V1.S4, V3.S4, V2.S4          // p = a1 + …
	VFMUL4(2, 2, 0)                    // p·x
	VMOV  V28.B16, V4.B16
	VFMLA V1.S4, V29.S4, V4.S4           // q = b4 + b6·x²
	VMOV  V27.B16, V3.B16
	VFMLA V1.S4, V4.S4, V3.S4
	VMOV  V26.B16, V4.B16
	VFMLA V1.S4, V3.S4, V4.S4          // q = b0 + …
	VFDIV4(5, 2, 4)                    // p / q
	VFCMGT4(7, 18, 6)                  // tiny > |x|
	VBSL16(7, 0, 5)                    // mask ? x : p/q
	VFCMGE4(12, 6, 10)                 // |x| >= 9
	VBSL16(12, 11, 7)                  // saturate to ±1
	VST1.P [V12.S4], 16(R2)
	SUBS $1, R3, R3
	BNE  tloop
tdone:
	RET

// ---------------------------------------------------------------------------
// func logNEON(x, z *float32, n int)      (n % 4 == 0)
//
// Cephes logf, see genericLog. Constants in V8..V31.
// ---------------------------------------------------------------------------
TEXT ·logNEON(SB), NOSPLIT, $0-24
	MOVD x+0(FP), R0
	MOVD z+8(FP), R2
	MOVD n+16(FP), R3
	MOVD $0x00800000, R4 // minnorm
	VDUP R4, V8.S4
	MOVD $0x3F3504F3, R4 // sqrthf
	VDUP R4, V9.S4
	MOVD $0x3F800000, R4 // one
	VDUP R4, V10.S4
	MOVD $0x3F000000, R4 // half
	VDUP R4, V11.S4
	MOVD $0x3D9021BB, R4 // p0
	VDUP R4, V12.S4
	MOVD $0xBDEBD1B8, R4 // p1
	VDUP R4, V13.S4
	MOVD $0x3DEF251A, R4 // p2
	VDUP R4, V14.S4
	MOVD $0xBDFE5D4F, R4 // p3
	VDUP R4, V15.S4
	MOVD $0x3E11E9BF, R4 // p4
	VDUP R4, V16.S4
	MOVD $0xBE2AAE50, R4 // p5
	VDUP R4, V17.S4
	MOVD $0x3E4CCEAC, R4 // p6
	VDUP R4, V18.S4
	MOVD $0xBE7FFFFC, R4 // p7
	VDUP R4, V19.S4
	MOVD $0x3EAAAAAA, R4 // p8
	VDUP R4, V20.S4
	MOVD $0xB95E8083, R4 // ln2lo
	VDUP R4, V21.S4
	MOVD $0x3F318000, R4 // ln2hi
	VDUP R4, V22.S4
	MOVD $0x007FFFFF, R4 // mant
	VDUP R4, V23.S4
	MOVD $0x0000007E, R4 // e126
	VDUP R4, V24.S4
	MOVD $0x7F800000, R4 // inf
	VDUP R4, V25.S4
	MOVD $0xFF800000, R4 // ninf
	VDUP R4, V26.S4
	MOVD $0x7FC00000, R4 // nan
	VDUP R4, V27.S4
	VEOR V28.B16, V28.B16, V28.B16 // 0
	LSR  $2, R3, R3
	CBZ  R3, ldone
lloop:
	VLD1.P 16(R0), [V0.S4]
	VFMAX4(1, 0, 8)                 // xc = max(x, min normal)
	VUSHR $23, V1.S4, V2.S4            // exponent field
	VSUB  V24.S4, V2.S4, V2.S4        // e = field - 126
	VSCVTF4(2, 2)                      // e as float
	VAND  V23.B16, V1.B16, V3.B16      // mantissa bits
	VORR  V11.B16, V3.B16, V3.B16      // m in [0.5, 1)
	VFCMGT4(4, 9, 3)                 // √½ > m
	VAND  V4.B16, V3.B16, V5.B16       // m or 0
	VAND  V4.B16, V10.B16, V6.B16       // 1 or 0
	VFSUB4(2, 2, 6)                    // e -= 1 where m < √½
	VFSUB4(3, 3, 10)                    // m -= 1
	VFADD4(3, 3, 5)                    // m += m where m < √½
	VFMUL4(7, 3, 3)                    // z = m²
	VMOV  V13.B16, V4.B16
	VFMLA V3.S4, V12.S4, V4.S4           // y = p1 + p0·m
	VMOV  V14.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4
	VMOV  V15.B16, V4.B16
	VFMLA V3.S4, V5.S4, V4.S4
	VMOV  V16.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4
	VMOV  V17.B16, V4.B16
	VFMLA V3.S4, V5.S4, V4.S4
	VMOV  V18.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4
	VMOV  V19.B16, V4.B16
	VFMLA V3.S4, V5.S4, V4.S4
	VMOV  V20.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4          // y = p8 + …
	VFMUL4(5, 5, 3)                    // · m
	VFMUL4(5, 5, 7)                    // · z
	VFMLA V2.S4, V21.S4, V5.S4        // += e·ln2lo
	VFMLS V7.S4, V11.S4, V5.S4         // -= ½z
	VFADD4(5, 5, 3)                    // + m
	VFMLA V2.S4, V22.S4, V5.S4        // += e·ln2hi
	VFCMEQ4(4, 0, 28)                   // x == 0 → -Inf
	VBSL16(4, 26, 5)
	VFCMGT4(6, 28, 0)                   // 0 > x → NaN
	VBSL16(6, 27, 4)
	VFCMEQ4(7, 0, 25)                    // x == +Inf → +Inf
	VBSL16(7, 25, 6)
	VST1.P [V7.S4], 16(R2)
	SUBS $1, R3, R3
	BNE  lloop
ldone:
	RET
