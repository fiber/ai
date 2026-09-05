#include "textflag.h"

// func packRows4(dst, src *float32, rs, pb, mr int)
//
// Four rows of A become four consecutive floats per k in the panel:
// load one 4-float vector from each row, transpose the 4×4 block with
// zip pairs, store the four columns at dst + (p+q)*mr.
TEXT ·packRows4(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD rs+16(FP), R2
	MOVD pb+24(FP), R3
	MOVD mr+32(FP), R4
	LSL  $2, R2, R2            // row stride in bytes
	LSL  $2, R4, R4            // panel stride in bytes
	ADD  R1, R2, R5            // row 1
	ADD  R5, R2, R6            // row 2
	ADD  R6, R2, R7            // row 3
	LSR  $2, R3, R3            // blocks of 4 columns
	CBZ  R3, done
loop:
	VLD1.P 16(R1), [V0.S4]
	VLD1.P 16(R5), [V1.S4]
	VLD1.P 16(R6), [V2.S4]
	VLD1.P 16(R7), [V3.S4]
	VZIP1 V1.S4, V0.S4, V4.S4  // r0[0] r1[0] r0[1] r1[1]
	VZIP2 V1.S4, V0.S4, V5.S4  // r0[2] r1[2] r0[3] r1[3]
	VZIP1 V3.S4, V2.S4, V6.S4  // r2[0] r3[0] r2[1] r3[1]
	VZIP2 V3.S4, V2.S4, V7.S4  // r2[2] r3[2] r2[3] r3[3]
	VZIP1 V6.D2, V4.D2, V16.D2 // column 0: r0[0] r1[0] r2[0] r3[0]
	VZIP2 V6.D2, V4.D2, V17.D2 // column 1
	VZIP1 V7.D2, V5.D2, V18.D2 // column 2
	VZIP2 V7.D2, V5.D2, V19.D2 // column 3
	VST1 [V16.S4], (R0)
	ADD  R4, R0, R0
	VST1 [V17.S4], (R0)
	ADD  R4, R0, R0
	VST1 [V18.S4], (R0)
	ADD  R4, R0, R0
	VST1 [V19.S4], (R0)
	ADD  R4, R0, R0
	SUBS $1, R3, R3
	BNE  loop
done:
	RET
