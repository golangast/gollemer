//go:build amd64

#include "textflag.h"

// func vecDotAVX2(a, b []float32) float32
// Stack frame: 0 bytes, 56 bytes arguments
TEXT ·vecDotAVX2(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), AX      // AX = ptr to a
    MOVQ a_len+8(FP), CX       // CX = len(a)
    MOVQ b_base+24(FP), BX      // BX = ptr to b

    VZEROUPPER
    VXORPS Y0, Y0, Y0          // Accumulator 0
    VXORPS Y6, Y6, Y6          // Accumulator 1
    VXORPS Y7, Y7, Y7          // Accumulator 2
    VXORPS Y8, Y8, Y8          // Accumulator 3

    CMPQ CX, $32
    JL loop8_prep

loop32:
    VMOVUPS (AX), Y1
    VMOVUPS 32(AX), Y3
    VMOVUPS 64(AX), Y4
    VMOVUPS 96(AX), Y5

    VFMADD231PS (BX), Y1, Y0
    VFMADD231PS 32(BX), Y3, Y6
    VFMADD231PS 64(BX), Y4, Y7
    VFMADD231PS 96(BX), Y5, Y8

    ADDQ $128, AX
    ADDQ $128, BX
    SUBQ $32, CX
    CMPQ CX, $32
    JGE loop32

    // Combine accumulators back into Y0
    VADDPS Y6, Y0, Y0
    VADDPS Y7, Y0, Y0
    VADDPS Y8, Y0, Y0

loop8_prep:
    CMPQ CX, $8
    JL tail_loop

loop8:
    VMOVUPS (AX), Y1           // Load 8 float32s from a
    VMOVUPS (BX), Y2           // Load 8 float32s from b
    VFMADD231PS Y1, Y2, Y0    // Y0 += Y1 * Y2 (Fused Multiply-Add)

    ADDQ $32, AX               // Advance ptr a by 32 bytes (8 floats)
    ADDQ $32, BX               // Advance ptr b by 32 bytes
    SUBQ $8, CX                // Remaining count -= 8
    CMPQ CX, $8
    JGE loop8

tail_loop:
    // Extract sum of 8 floats in Y0 down to X0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    // Process remaining elements < 8
    TESTQ CX, CX
    JZ done

tail_single:
    MOVSS (AX), X1
    MULSS (BX), X1
    ADDSS X1, X0
    ADDQ $4, AX
    ADDQ $4, BX
    DECQ CX
    JNZ tail_single

done:
    MOVSS X0, ret+48(FP)
    VZEROUPPER
    RET

// func vecAddAVX2(dst, src []float32)
TEXT ·vecAddAVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), AX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), BX

    VZEROUPPER

    CMPQ CX, $32
    JL loop8_prep_add

loop32_add:
    VMOVUPS (AX), Y0
    VMOVUPS 32(AX), Y2
    VMOVUPS 64(AX), Y3
    VMOVUPS 96(AX), Y4

    VMOVUPS (BX), Y1
    VMOVUPS 32(BX), Y5
    VMOVUPS 64(BX), Y6
    VMOVUPS 96(BX), Y7

    VADDPS Y0, Y1, Y0
    VADDPS Y2, Y5, Y2
    VADDPS Y3, Y6, Y3
    VADDPS Y4, Y7, Y4

    VMOVUPS Y0, (AX)
    VMOVUPS Y2, 32(AX)
    VMOVUPS Y3, 64(AX)
    VMOVUPS Y4, 96(AX)

    ADDQ $128, AX
    ADDQ $128, BX
    SUBQ $32, CX
    CMPQ CX, $32
    JGE loop32_add

loop8_prep_add:
    CMPQ CX, $8
    JL tail_add

loop8_add:
    VMOVUPS (AX), Y0
    VMOVUPS (BX), Y1
    VADDPS Y0, Y1, Y0
    VMOVUPS Y0, (AX)

    ADDQ $32, AX
    ADDQ $32, BX
    SUBQ $8, CX
    CMPQ CX, $8
    JGE loop8_add

tail_add:
    TESTQ CX, CX
    JZ add_done

tail_add_single:
    MOVSS (AX), X0
    ADDSS (BX), X0
    MOVSS X0, (AX)
    ADDQ $4, AX
    ADDQ $4, BX
    DECQ CX
    JNZ tail_add_single

add_done:
    VZEROUPPER
    RET

// func vecSubAVX2(a, b, res []float32)
TEXT ·vecSubAVX2(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), AX
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), BX
    MOVQ res_base+48(FP), DX

    VZEROUPPER

    CMPQ CX, $32
    JL loop8_prep_sub

loop32_sub:
    VMOVUPS (AX), Y0
    VMOVUPS 32(AX), Y2
    VMOVUPS 64(AX), Y3
    VMOVUPS 96(AX), Y4

    VMOVUPS (BX), Y1
    VMOVUPS 32(BX), Y5
    VMOVUPS 64(BX), Y6
    VMOVUPS 96(BX), Y7

    VSUBPS Y1, Y0, Y0
    VSUBPS Y5, Y2, Y2
    VSUBPS Y6, Y3, Y3
    VSUBPS Y7, Y4, Y4

    VMOVUPS Y0, (DX)
    VMOVUPS Y2, 32(DX)
    VMOVUPS Y3, 64(DX)
    VMOVUPS Y4, 96(DX)

    ADDQ $128, AX
    ADDQ $128, BX
    ADDQ $128, DX
    SUBQ $32, CX
    CMPQ CX, $32
    JGE loop32_sub

loop8_prep_sub:
    CMPQ CX, $8
    JL tail_sub

loop8_sub:
    VMOVUPS (AX), Y0
    VMOVUPS (BX), Y1
    VSUBPS Y1, Y0, Y0
    VMOVUPS Y0, (DX)

    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, DX
    SUBQ $8, CX
    CMPQ CX, $8
    JGE loop8_sub

tail_sub:
    TESTQ CX, CX
    JZ sub_done

tail_sub_single:
    MOVSS (AX), X0
    MOVSS (BX), X1
    SUBSS X1, X0
    MOVSS X0, (DX)
    ADDQ $4, AX
    ADDQ $4, BX
    ADDQ $4, DX
    DECQ CX
    JNZ tail_sub_single

sub_done:
    VZEROUPPER
    RET

// func vecMulAVX2(a, b, res []float32)
TEXT ·vecMulAVX2(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), AX
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), BX
    MOVQ res_base+48(FP), DX

    VZEROUPPER

    CMPQ CX, $32
    JL loop8_prep_mul

loop32_mul:
    VMOVUPS (AX), Y0
    VMOVUPS 32(AX), Y2
    VMOVUPS 64(AX), Y3
    VMOVUPS 96(AX), Y4

    VMOVUPS (BX), Y1
    VMOVUPS 32(BX), Y5
    VMOVUPS 64(BX), Y6
    VMOVUPS 96(BX), Y7

    VMULPS Y1, Y0, Y0
    VMULPS Y5, Y2, Y2
    VMULPS Y6, Y3, Y3
    VMULPS Y7, Y4, Y4

    VMOVUPS Y0, (DX)
    VMOVUPS Y2, 32(DX)
    VMOVUPS Y3, 64(DX)
    VMOVUPS Y4, 96(DX)

    ADDQ $128, AX
    ADDQ $128, BX
    ADDQ $128, DX
    SUBQ $32, CX
    CMPQ CX, $32
    JGE loop32_mul

loop8_prep_mul:
    CMPQ CX, $8
    JL tail_mul

loop8_mul:
    VMOVUPS (AX), Y0
    VMOVUPS (BX), Y1
    VMULPS Y1, Y0, Y0
    VMOVUPS Y0, (DX)

    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, DX
    SUBQ $8, CX
    CMPQ CX, $8
    JGE loop8_mul

tail_mul:
    TESTQ CX, CX
    JZ mul_done

tail_mul_single:
    MOVSS (AX), X0
    MOVSS (BX), X1
    MULSS X1, X0
    MOVSS X0, (DX)
    ADDQ $4, AX
    ADDQ $4, BX
    ADDQ $4, DX
    DECQ CX
    JNZ tail_mul_single

mul_done:
    VZEROUPPER
    RET

// func vecSoftmaxBackwardRowAVX2(p, dp, out []float32)
TEXT ·vecSoftmaxBackwardRowAVX2(SB), NOSPLIT, $0-72
    MOVQ p_base+0(FP), AX
    MOVQ p_len+8(FP), CX
    MOVQ dp_base+24(FP), BX
    MOVQ out_base+48(FP), DX

    VZEROUPPER
    VXORPS Y0, Y0, Y0
    VXORPS Y10, Y10, Y10
    VXORPS Y11, Y11, Y11
    VXORPS Y12, Y12, Y12

    MOVQ AX, SI
    MOVQ BX, DI
    MOVQ CX, R8

    CMPQ R8, $32
    JL loop8_prep_dot

loop32_dot:
    VMOVUPS (SI), Y1
    VMOVUPS 32(SI), Y3
    VMOVUPS 64(SI), Y4
    VMOVUPS 96(SI), Y5

    VMOVUPS (DI), Y2
    VMOVUPS 32(DI), Y6
    VMOVUPS 64(DI), Y7
    VMOVUPS 96(DI), Y8

    VFMADD231PS Y1, Y2, Y0
    VFMADD231PS Y3, Y6, Y10
    VFMADD231PS Y4, Y7, Y11
    VFMADD231PS Y5, Y8, Y12

    ADDQ $128, SI
    ADDQ $128, DI
    SUBQ $32, R8
    CMPQ R8, $32
    JGE loop32_dot

    // Combine accumulators back into Y0
    VADDPS Y10, Y0, Y0
    VADDPS Y11, Y0, Y0
    VADDPS Y12, Y0, Y0

loop8_prep_dot:
    CMPQ R8, $8
    JL tail_dot

loop8_dot:
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VFMADD231PS Y1, Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $8, R8
    CMPQ R8, $8
    JGE loop8_dot

tail_dot:
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    TESTQ R8, R8
    JZ dot_done

tail_dot_single:
    MOVSS (SI), X1
    MULSS (DI), X1
    ADDSS X1, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ R8
    JNZ tail_dot_single

dot_done:
    // Broadcast dot to Y2
    VSHUFPS $0, X0, X0, X0
    VMOVAPS X0, X2
    VINSERTF128 $1, X0, Y2, Y2

    // Now compute out[i] = p[i] * (dp[i] - dot)
    CMPQ CX, $32
    JL loop8_prep_out

loop32_out:
    VMOVUPS (BX), Y1
    VMOVUPS 32(BX), Y4
    VMOVUPS 64(BX), Y6
    VMOVUPS 96(BX), Y8

    VSUBPS Y2, Y1, Y1
    VSUBPS Y2, Y4, Y4
    VSUBPS Y2, Y6, Y6
    VSUBPS Y2, Y8, Y8

    VMOVUPS (AX), Y0
    VMOVUPS 32(AX), Y3
    VMOVUPS 64(AX), Y5
    VMOVUPS 96(AX), Y7

    VMULPS Y1, Y0, Y0
    VMULPS Y4, Y3, Y3
    VMULPS Y6, Y5, Y5
    VMULPS Y8, Y7, Y7

    VMOVUPS Y0, (DX)
    VMOVUPS Y3, 32(DX)
    VMOVUPS Y5, 64(DX)
    VMOVUPS Y7, 96(DX)

    ADDQ $128, AX
    ADDQ $128, BX
    ADDQ $128, DX
    SUBQ $32, CX
    CMPQ CX, $32
    JGE loop32_out

loop8_prep_out:
    CMPQ CX, $8
    JL tail_out

loop8_out:
    VMOVUPS (AX), Y0
    VMOVUPS (BX), Y1
    VSUBPS Y2, Y1, Y1
    VMULPS Y1, Y0, Y0
    VMOVUPS Y0, (DX)

    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, DX
    SUBQ $8, CX
    CMPQ CX, $8
    JGE loop8_out

tail_out:
    TESTQ CX, CX
    JZ out_done

tail_out_single:
    MOVSS (AX), X0
    MOVSS (BX), X1
    VSUBSS X2, X1, X1 // X1 = dp[i] - dot
    VMULSS X1, X0, X0 // X0 = p[i] * (dp[i] - dot)
    MOVSS X0, (DX)
    ADDQ $4, AX
    ADDQ $4, BX
    ADDQ $4, DX
    DECQ CX
    JNZ tail_out_single

out_done:
    VZEROUPPER
    RET
