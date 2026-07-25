//go:build arm64

#include "textflag.h"

// func vecDotNEON(a, b []float32) float32
TEXT ·vecDotNEON(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD b_base+24(FP), R2

    VMOV S0, V0.S4             // Clear V0
    VMOV S0, V1.S4             // Clear V1

    CMP $4, R1
    BLT tail_neon

loop4_neon:
    VLD1.P 16(R0), [V2.S4]     // Load 4 floats from a, auto-increment
    VLD1.P 16(R2), [V3.S4]     // Load 4 floats from b, auto-increment
    VFMLA V2.S4, V3.S4, V0.S4  // V0 += V2 * V3 (Vector FMA)

    SUB $4, R1
    CMP $4, R1
    BGE loop4_neon

tail_neon:
    // Horizontal addition across 4 lanes of V0
    FADDP V0.S2, V0.S2, V0.S2
    FADDP V0.S2, V0.S2, V0.S2

    CBZ R1, neon_done

tail_single_neon:
    FMOVS (R0), F1
    FMOVS (R2), F2
    FMULS F1, F2, F1
    FADDS F1, F0, F0
    ADD $4, R0
    ADD $4, R2
    SUB $1, R1
    CBNZ R1, tail_single_neon

neon_done:
    FMOVS F0, ret+48(FP)
    RET

// func vecAddNEON(dst, src []float32)
// This wasn't fully provided in the snippet, but I will implement it for completeness
TEXT ·vecAddNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD src_base+24(FP), R2

    CMP $4, R1
    BLT tail_add_neon

loop4_add_neon:
    VLD1 16(R0), [V0.S4]       // Load 4 floats from dst (no increment yet)
    VLD1.P 16(R2), [V1.S4]     // Load 4 floats from src, increment R2
    FADD V1.S4, V0.S4, V0.S4   // V0 = V0 + V1
    VST1.P [V0.S4], 16(R0)     // Store and increment R0

    SUB $4, R1
    CMP $4, R1
    BGE loop4_add_neon

tail_add_neon:
    CBZ R1, add_neon_done

tail_add_single_neon:
    FMOVS (R0), F0
    FMOVS (R2), F1
    FADDS F1, F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R2
    SUB $1, R1
    CBNZ R1, tail_add_single_neon

add_neon_done:
    RET

// func vecSubNEON(a, b, res []float32)
TEXT ·vecSubNEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD b_base+24(FP), R2
    MOVD res_base+48(FP), R3

    CMP $4, R1
    BLT tail_sub_neon

loop4_sub_neon:
    VLD1.P 16(R0), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    FSUB V1.S4, V0.S4, V0.S4
    VST1.P [V0.S4], 16(R3)

    SUB $4, R1
    CMP $4, R1
    BGE loop4_sub_neon

tail_sub_neon:
    CBZ R1, sub_neon_done

tail_sub_single_neon:
    FMOVS (R0), F0
    FMOVS (R2), F1
    FSUBS F1, F0, F0
    FMOVS F0, (R3)
    ADD $4, R0
    ADD $4, R2
    ADD $4, R3
    SUB $1, R1
    CBNZ R1, tail_sub_single_neon

sub_neon_done:
    RET

// func vecMulNEON(a, b, res []float32)
TEXT ·vecMulNEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD b_base+24(FP), R2
    MOVD res_base+48(FP), R3

    CMP $4, R1
    BLT tail_mul_neon

loop4_mul_neon:
    VLD1.P 16(R0), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    FMUL V1.S4, V0.S4, V0.S4
    VST1.P [V0.S4], 16(R3)

    SUB $4, R1
    CMP $4, R1
    BGE loop4_mul_neon

tail_mul_neon:
    CBZ R1, mul_neon_done

tail_mul_single_neon:
    FMOVS (R0), F0
    FMOVS (R2), F1
    FMULS F1, F0, F0
    FMOVS F0, (R3)
    ADD $4, R0
    ADD $4, R2
    ADD $4, R3
    SUB $1, R1
    CBNZ R1, tail_mul_single_neon

mul_neon_done:
    RET

// func vecSoftmaxBackwardRowNEON(p, dp, out []float32)
TEXT ·vecSoftmaxBackwardRowNEON(SB), NOSPLIT, $0-72
    MOVD p_base+0(FP), R0
    MOVD p_len+8(FP), R1
    MOVD dp_base+24(FP), R2
    MOVD out_base+48(FP), R3

    MOVD R0, R4
    MOVD R2, R5
    MOVD R1, R6

    VMOV S0, V0.S4
    VMOV S0, V1.S4

    CMP $4, R6
    BLT tail_dot_neon

loop4_dot_neon:
    VLD1.P 16(R4), [V2.S4]
    VLD1.P 16(R5), [V3.S4]
    VFMLA V2.S4, V3.S4, V0.S4

    SUB $4, R6
    CMP $4, R6
    BGE loop4_dot_neon

tail_dot_neon:
    FADDP V0.S2, V0.S2, V0.S2
    FADDP V0.S2, V0.S2, V0.S2

    CBZ R6, dot_neon_done

tail_dot_single_neon:
    FMOVS (R4), F1
    FMOVS (R5), F2
    FMULS F1, F2, F1
    FADDS F1, F0, F0
    ADD $4, R4
    ADD $4, R5
    SUB $1, R6
    CBNZ R6, tail_dot_single_neon

dot_neon_done:
    // broadcast dot (F0) into V4.S4
    DUP F0, V4.S4

    CMP $4, R1
    BLT tail_out_neon

loop4_out_neon:
    VLD1.P 16(R0), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    FSUB V4.S4, V1.S4, V1.S4
    FMUL V1.S4, V0.S4, V0.S4
    VST1.P [V0.S4], 16(R3)

    SUB $4, R1
    CMP $4, R1
    BGE loop4_out_neon

tail_out_neon:
    CBZ R1, out_neon_done

tail_out_single_neon:
    FMOVS (R0), F0
    FMOVS (R2), F1
    FSUBS F4, F1, F1   // Wait, F4 holds the dot product? In ARM64, V4.S4 is mapped to Q4/D4/S4. F0 was DUP'd to V4.S4, so F4 contains dot product!
    FMULS F1, F0, F0
    FMOVS F0, (R3)
    ADD $4, R0
    ADD $4, R2
    ADD $4, R3
    SUB $1, R1
    CBNZ R1, tail_out_single_neon

out_neon_done:
    RET
