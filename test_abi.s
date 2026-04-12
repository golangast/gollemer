	.file	"test_abi.c"
	.text
	.p2align 4
	.globl	test_call
	.type	test_call, @function
test_call:
.LFB0:
	.cfi_startproc
	endbr64
	subq	$72, %rsp
	.cfi_def_cfa_offset 80
	movl	$2, %esi
	movl	$1, %edi
	movdqa	.LC0(%rip), %xmm0
	movq	$0, 32(%rsp)
	movl	$2, 40(%rsp)
	movdqa	32(%rsp), %xmm1
	movaps	%xmm0, 48(%rsp)
	movdqa	48(%rsp), %xmm2
	movups	%xmm1, (%rsp)
	movups	%xmm2, 16(%rsp)
	call	call_rust@PLT
	addq	$72, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE0:
	.size	test_call, .-test_call
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC0:
	.quad	4660
	.quad	22136
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
