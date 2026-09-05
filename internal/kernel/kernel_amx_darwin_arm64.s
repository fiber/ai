#include "textflag.h"

// Apple AMX instruction words: 0x201000 | op<<5 | Xn, the operand
// descriptor always in R10. SET/CLR take an immediate and are preceded by
// three NOPs as in the reference header (github.com/corsix/amx).

#define AMX_NOP3 WORD $0xD503201F; WORD $0xD503201F; WORD $0xD503201F
#define AMX_SET  WORD $0x00201220
#define AMX_CLR  WORD $0x00201221
#define AMX_LDX   WORD $(0x00201000 | (0<<5) | 10)
#define AMX_LDY   WORD $(0x00201000 | (1<<5) | 10)
#define AMX_LDZ   WORD $(0x00201000 | (4<<5) | 10)
#define AMX_STZ   WORD $(0x00201000 | (5<<5) | 10)
#define AMX_FMA32 WORD $(0x00201000 | (12<<5) | 10)

// fma descriptor for slot s (registers 2s,2s+1), tile t = 2a+b:
// yoff = s*128 + a*64, xoff = s*128 + b*64, zrow = t
#define FMA(s, a, b) MOVD $(((s)*128 + (a)*64) | (((s)*128 + (b)*64)<<10) | (((a)*2+(b))<<20)), R10; AMX_FMA32
#define FMA4(s) FMA(s,0,0); FMA(s,0,1); FMA(s,1,0); FMA(s,1,1)
// FMAZ ignores the Z input (bit 27): the first k-step of an overwriting tile
#define FMAZ(s, a, b) MOVD $((1<<27) | ((s)*128 + (a)*64) | (((s)*128 + (b)*64)<<10) | (((a)*2+(b))<<20)), R10; AMX_FMA32
#define FMA4Z(s) FMAZ(s,0,0); FMAZ(s,0,1); FMAZ(s,1,0); FMAZ(s,1,1)
// pair load of A (R1) into Y slot s and B (R2) into X slot s, advancing both pointers
#define LOAD(s) MOVD $(((2*(s))<<56) | (1<<62)), R11; ORR R1, R11, R10; AMX_LDY; ORR R2, R11, R10; AMX_LDX; ADD $128, R1, R1; ADD $128, R2, R2

// func amxSet()
TEXT ·amxSet(SB), NOSPLIT, $0-0
	AMX_NOP3
	AMX_SET
	RET

// func amxClr()
TEXT ·amxClr(SB), NOSPLIT, $0-0
	AMX_NOP3
	AMX_CLR
	RET

// func gemmAMX(k int, a, b, c *float32, ldc int)
//
// C[32×32] += A[k×32 packed] · B[k×32 packed], C row-major with ldc
// floats. Z holds four 16×16 f32 tiles t = 2a+b (A half a, B half b): C
// row j, column half h lives in Z row 4*(j&15) + 2*(j>>4) + h. The k loop
// is software-pipelined over four X/Y register slots so the loads of
// step k+4 overlap the outer products of step k (the coprocessor executes
// in order; without this the kernel ran at a third of the speed). The
// thread must have executed AMX_SET (see amxBegin).
TEXT ·gemmAMX(SB), NOSPLIT, $0-40
	MOVD $1, R19
	B    ·gemmAMXBody(SB)

// gemmZeroAMX overwrites C (β = 0, k > 0): no Z load, and the first k-step
// runs its outer products with the Z input ignored.
TEXT ·gemmZeroAMX(SB), NOSPLIT, $0-40
	MOVD $0, R19
	B    ·gemmAMXBody(SB)

TEXT ·gemmAMXBody(SB), NOSPLIT, $0-40
	MOVD k+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	MOVD c+24(FP), R3
	MOVD ldc+32(FP), R4
	LSL  $2, R4, R4
	CBNZ R19, loadc
	// overwrite: peel step 0 with Z ignored, then continue accumulating
	LOAD(0)
	FMA4Z(0)
	SUB  $1, R0, R0
	B    kstart
loadc:
	MOVD $0, R5
	MOVD R3, R6
ldz:
	AND  $15, R5, R7
	LSL  $2, R7, R7
	LSR  $4, R5, R8
	LSL  $1, R8, R8
	ADD  R8, R7, R7
	LSL  $56, R7, R9
	ORR  R6, R9, R10
	AMX_LDZ
	ADD  $1, R7, R7
	LSL  $56, R7, R9
	ADD  $64, R6, R11
	ORR  R11, R9, R10
	AMX_LDZ
	ADD  R4, R6, R6
	ADD  $1, R5, R5
	CMP  $32, R5
	BNE  ldz
kstart:
	// pipelined main loop: 4 steps per iteration; loads for slot s of the
	// next block are issued right after the FMAs of slot s.
	CMP  $8, R0
	BLT  tail
	LOAD(0)
	LOAD(1)
	LOAD(2)
	LOAD(3)
	SUB  $4, R0, R0
kloop:
	FMA4(0)
	LOAD(0)
	FMA4(1)
	LOAD(1)
	FMA4(2)
	LOAD(2)
	FMA4(3)
	LOAD(3)
	SUB  $4, R0, R0
	CMP  $4, R0
	BGE  kloop
	// drain the four preloaded slots
	FMA4(0)
	FMA4(1)
	FMA4(2)
	FMA4(3)
tail:
	CBZ  R0, store
tloop:
	LOAD(0)
	FMA4(0)
	SUBS $1, R0, R0
	BNE  tloop
store:
	MOVD $0, R5
	MOVD R3, R6
stz:
	AND  $15, R5, R7
	LSL  $2, R7, R7
	LSR  $4, R5, R8
	LSL  $1, R8, R8
	ADD  R8, R7, R7
	LSL  $56, R7, R9
	ORR  R6, R9, R10
	AMX_STZ
	ADD  $1, R7, R7
	LSL  $56, R7, R9
	ADD  $64, R6, R11
	ORR  R11, R9, R10
	AMX_STZ
	ADD  R4, R6, R6
	ADD  $1, R5, R5
	CMP  $32, R5
	BNE  stz
	RET
