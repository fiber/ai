#include "textflag.h"

// SME (Scalable Matrix Extension) instruction words, from the LLVM MC
// tests (llvm/test/MC/AArch64/SME, SVE). Streaming SVE vector length on
// the M4 is 512 bits: 16 floats per Z register, four 16×16 f32 ZA tiles.
#define SMSTART WORD $0xD503477F
#define SMSTOP  WORD $0xD503467F
#define ZERO_ZA WORD $0xC00800FF
#define PTRUE_P0_S WORD $0x2598E3E0
// ld1w {Zt.s}, p0/z, [Xn]              0xA540A000 | Rn<<5 | Zt
#define LD1W(Zt, Rn) WORD $(0xA540A000 | ((Rn)<<5) | (Zt))
// st1w {Zt.s}, p0, [Xn]                0xE540E000 | Rn<<5 | Zt
#define ST1W(Zt, Rn) WORD $(0xE540E000 | ((Rn)<<5) | (Zt))
// fmopa ZAda.s, p0/m, p0/m, Zn.s, Zm.s 0x80800000 | Zm<<16 | Zn<<5 | ZAda
#define FMOPA(ZAda, Zn, Zm) WORD $(0x80800000 | ((Zm)<<16) | ((Zn)<<5) | (ZAda))
// mova Zd.s, p0/m, ZAn h.s[w12, imm]   0xC0820000 | imm<<5 | ZAn<<7 | Zd   (Ws = w12)
#define MOVA_ROW(Zd, ZAn, imm) WORD $(0xC0820000 | ((imm)<<5) | ((ZAn)<<7) | (Zd))

// func outer(x, y, z *float32)
// z[16][16] = y[j] * x[i]  (fmopa: ZA[row j][col i] += Zn[j] * Zm[i])
TEXT ·outer(SB), NOSPLIT, $0-24
	MOVD x+0(FP), R0
	MOVD y+8(FP), R1
	MOVD z+16(FP), R2
	SMSTART
	PTRUE_P0_S
	ZERO_ZA
	LD1W(0, 0)                 // z0 = x
	LD1W(1, 1)                 // z1 = y
	FMOPA(0, 1, 0)             // za0 += z1 ⊗ z0
	MOVD $0, R12
	MOVA_ROW(2, 0, 0); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 1); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 2); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 3); ST1W(2, 2); ADD $64, R2, R2
	MOVD $4, R12
	MOVA_ROW(2, 0, 0); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 1); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 2); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 3); ST1W(2, 2); ADD $64, R2, R2
	MOVD $8, R12
	MOVA_ROW(2, 0, 0); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 1); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 2); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 3); ST1W(2, 2); ADD $64, R2, R2
	MOVD $12, R12
	MOVA_ROW(2, 0, 0); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 1); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 2); ST1W(2, 2); ADD $64, R2, R2
	MOVA_ROW(2, 0, 3); ST1W(2, 2); ADD $64, R2, R2
	SMSTOP
	RET

// func tile(k int, a, b *float32)
// k steps of the 32×32 kernel body (four fmopa per step over za0..za3),
// results discarded: measures the outer-product throughput alone.
TEXT ·tile(SB), NOSPLIT, $0-24
	MOVD k+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	SMSTART
	PTRUE_P0_S
	ZERO_ZA
	ADD  $64, R1, R3           // second half of the A column
	ADD  $64, R2, R4           // second half of the B row
	CBZ  R0, done
loop:
	LD1W(0, 1)
	LD1W(1, 3)
	LD1W(2, 2)
	LD1W(3, 4)
	FMOPA(0, 0, 2)
	FMOPA(1, 0, 3)
	FMOPA(2, 1, 2)
	FMOPA(3, 1, 3)
	ADD  $128, R1, R1
	ADD  $128, R3, R3
	ADD  $128, R2, R2
	ADD  $128, R4, R4
	SUBS $1, R0, R0
	BNE  loop
done:
	SMSTOP
	RET
