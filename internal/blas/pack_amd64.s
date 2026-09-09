#include "textflag.h"

// Store masks: entry r has its first r lanes set.
DATA ·packMask8+0(SB)/4, $0
DATA ·packMask8+4(SB)/4, $0
DATA ·packMask8+8(SB)/4, $0
DATA ·packMask8+12(SB)/4, $0
DATA ·packMask8+16(SB)/4, $0
DATA ·packMask8+20(SB)/4, $0
DATA ·packMask8+24(SB)/4, $0
DATA ·packMask8+28(SB)/4, $0
DATA ·packMask8+32(SB)/4, $0xFFFFFFFF
DATA ·packMask8+36(SB)/4, $0
DATA ·packMask8+40(SB)/4, $0
DATA ·packMask8+44(SB)/4, $0
DATA ·packMask8+48(SB)/4, $0
DATA ·packMask8+52(SB)/4, $0
DATA ·packMask8+56(SB)/4, $0
DATA ·packMask8+60(SB)/4, $0
DATA ·packMask8+64(SB)/4, $0xFFFFFFFF
DATA ·packMask8+68(SB)/4, $0xFFFFFFFF
DATA ·packMask8+72(SB)/4, $0
DATA ·packMask8+76(SB)/4, $0
DATA ·packMask8+80(SB)/4, $0
DATA ·packMask8+84(SB)/4, $0
DATA ·packMask8+88(SB)/4, $0
DATA ·packMask8+92(SB)/4, $0
DATA ·packMask8+96(SB)/4, $0xFFFFFFFF
DATA ·packMask8+100(SB)/4, $0xFFFFFFFF
DATA ·packMask8+104(SB)/4, $0xFFFFFFFF
DATA ·packMask8+108(SB)/4, $0
DATA ·packMask8+112(SB)/4, $0
DATA ·packMask8+116(SB)/4, $0
DATA ·packMask8+120(SB)/4, $0
DATA ·packMask8+124(SB)/4, $0
DATA ·packMask8+128(SB)/4, $0xFFFFFFFF
DATA ·packMask8+132(SB)/4, $0xFFFFFFFF
DATA ·packMask8+136(SB)/4, $0xFFFFFFFF
DATA ·packMask8+140(SB)/4, $0xFFFFFFFF
DATA ·packMask8+144(SB)/4, $0
DATA ·packMask8+148(SB)/4, $0
DATA ·packMask8+152(SB)/4, $0
DATA ·packMask8+156(SB)/4, $0
DATA ·packMask8+160(SB)/4, $0xFFFFFFFF
DATA ·packMask8+164(SB)/4, $0xFFFFFFFF
DATA ·packMask8+168(SB)/4, $0xFFFFFFFF
DATA ·packMask8+172(SB)/4, $0xFFFFFFFF
DATA ·packMask8+176(SB)/4, $0xFFFFFFFF
DATA ·packMask8+180(SB)/4, $0
DATA ·packMask8+184(SB)/4, $0
DATA ·packMask8+188(SB)/4, $0
DATA ·packMask8+192(SB)/4, $0xFFFFFFFF
DATA ·packMask8+196(SB)/4, $0xFFFFFFFF
DATA ·packMask8+200(SB)/4, $0xFFFFFFFF
DATA ·packMask8+204(SB)/4, $0xFFFFFFFF
DATA ·packMask8+208(SB)/4, $0xFFFFFFFF
DATA ·packMask8+212(SB)/4, $0xFFFFFFFF
DATA ·packMask8+216(SB)/4, $0
DATA ·packMask8+220(SB)/4, $0
DATA ·packMask8+224(SB)/4, $0xFFFFFFFF
DATA ·packMask8+228(SB)/4, $0xFFFFFFFF
DATA ·packMask8+232(SB)/4, $0xFFFFFFFF
DATA ·packMask8+236(SB)/4, $0xFFFFFFFF
DATA ·packMask8+240(SB)/4, $0xFFFFFFFF
DATA ·packMask8+244(SB)/4, $0xFFFFFFFF
DATA ·packMask8+248(SB)/4, $0xFFFFFFFF
DATA ·packMask8+252(SB)/4, $0
DATA ·packMask8+256(SB)/4, $0xFFFFFFFF
DATA ·packMask8+260(SB)/4, $0xFFFFFFFF
DATA ·packMask8+264(SB)/4, $0xFFFFFFFF
DATA ·packMask8+268(SB)/4, $0xFFFFFFFF
DATA ·packMask8+272(SB)/4, $0xFFFFFFFF
DATA ·packMask8+276(SB)/4, $0xFFFFFFFF
DATA ·packMask8+280(SB)/4, $0xFFFFFFFF
DATA ·packMask8+284(SB)/4, $0xFFFFFFFF
GLOBL ·packMask8(SB), RODATA|NOPTR, $288

// func packRows8(dst, src *float32, rs, pb, width, rows int)
//
// Per block of 8 columns: load the `rows` valid source rows (the rest
// zero), transpose 8×8 in registers (unpack, shuffle, perm2f128), store
// the eight panel rows under a lane mask of `rows` lanes.
TEXT ·packRows8(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ rs+16(FP), BX
	MOVQ pb+24(FP), CX
	MOVQ width+32(FP), DX
	MOVQ rows+40(FP), AX
	SHLQ $2, BX                // row stride in bytes
	SHLQ $2, DX                // panel row stride in bytes
	LEAQ (SI)(BX*1), R8        // row 1
	LEAQ (R8)(BX*1), R9        // row 2
	LEAQ (R9)(BX*1), R10       // row 3
	LEAQ (R10)(BX*1), R11      // row 4
	LEAQ (R11)(BX*1), R12      // row 5
	LEAQ (R12)(BX*1), R13      // row 6
	LEAQ (R13)(BX*1), R14      // row 7
	SHRQ $3, CX                // 8-column blocks
	JZ   done
	MOVQ AX, BX
	SHLQ $5, BX                // rows·32: offset of the mask entry
	LEAQ ·packMask8(SB), R15
	VMOVDQU (R15)(BX*1), Y15   // store mask: lanes < rows set
loop:
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VMOVUPS (SI), Y0
	CMPQ AX, $1
	JLE  transpose
	VMOVUPS (R8), Y1
	CMPQ AX, $2
	JLE  transpose
	VMOVUPS (R9), Y2
	CMPQ AX, $3
	JLE  transpose
	VMOVUPS (R10), Y3
	CMPQ AX, $4
	JLE  transpose
	VMOVUPS (R11), Y4
	CMPQ AX, $5
	JLE  transpose
	VMOVUPS (R12), Y5
	CMPQ AX, $6
	JLE  transpose
	VMOVUPS (R13), Y6
	CMPQ AX, $7
	JLE  transpose
	VMOVUPS (R14), Y7
transpose:
	VUNPCKLPS Y1, Y0, Y8       // t0 = r0[0] r1[0] r0[1] r1[1] | r0[4] r1[4] r0[5] r1[5]
	VUNPCKHPS Y1, Y0, Y9       // t1 = r0[2] r1[2] r0[3] r1[3] | ...
	VUNPCKLPS Y3, Y2, Y10      // t2
	VUNPCKHPS Y3, Y2, Y11      // t3
	VUNPCKLPS Y5, Y4, Y12      // t4
	VUNPCKHPS Y5, Y4, Y13      // t5
	VUNPCKLPS Y7, Y6, Y14      // t6
	VUNPCKHPS Y7, Y6, Y0       // t7
	VSHUFPS $0x44, Y10, Y8, Y1 // u0 = column 0 of rows 0-3 | column 4
	VSHUFPS $0xEE, Y10, Y8, Y2 // u1 = column 1 | 5
	VSHUFPS $0x44, Y11, Y9, Y3 // u2 = column 2 | 6
	VSHUFPS $0xEE, Y11, Y9, Y4 // u3 = column 3 | 7
	VSHUFPS $0x44, Y14, Y12, Y5 // u4 = rows 4-7, column 0 | 4
	VSHUFPS $0xEE, Y14, Y12, Y6 // u5
	VSHUFPS $0x44, Y0, Y13, Y7  // u6
	VSHUFPS $0xEE, Y0, Y13, Y8  // u7
	VPERM2F128 $0x20, Y5, Y1, Y9  // column 0
	VPERM2F128 $0x20, Y6, Y2, Y10 // column 1
	VPERM2F128 $0x20, Y7, Y3, Y11 // column 2
	VPERM2F128 $0x20, Y8, Y4, Y12 // column 3
	VPERM2F128 $0x31, Y5, Y1, Y13 // column 4
	VPERM2F128 $0x31, Y6, Y2, Y14 // column 5
	VPERM2F128 $0x31, Y7, Y3, Y0  // column 6
	VPERM2F128 $0x31, Y8, Y4, Y1  // column 7
	VMASKMOVPS Y9, Y15, (DI)
	ADDQ DX, DI
	VMASKMOVPS Y10, Y15, (DI)
	ADDQ DX, DI
	VMASKMOVPS Y11, Y15, (DI)
	ADDQ DX, DI
	VMASKMOVPS Y12, Y15, (DI)
	ADDQ DX, DI
	VMASKMOVPS Y13, Y15, (DI)
	ADDQ DX, DI
	VMASKMOVPS Y14, Y15, (DI)
	ADDQ DX, DI
	VMASKMOVPS Y0, Y15, (DI)
	ADDQ DX, DI
	VMASKMOVPS Y1, Y15, (DI)
	ADDQ DX, DI
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	ADDQ $32, R11
	ADDQ $32, R12
	ADDQ $32, R13
	ADDQ $32, R14
	DECQ CX
	JNZ  loop
done:
	VZEROUPPER
	RET
