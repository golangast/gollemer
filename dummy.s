#include "textflag.h"
TEXT ·vecDot(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), AX
    MOVSS (AX), X0
    MOVSS X0, ret+48(FP)
    RET
