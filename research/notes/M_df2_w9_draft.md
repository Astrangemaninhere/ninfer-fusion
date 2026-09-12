
# 用今天训练出来的草稿重造 artifact 后的接受率 (2026-09-10 15:46)

| run | accept (tok/round (%) ) | gen | finish | decode |
|---|---|---|---|---|
| old_zh | SERVE_FAILED | - | - | - |
| new_num | SERVE_FAILED | - | - | - |
| old_num | 1.00tok/round (0.0%)  | 2 | stop_token | 29.1tok/s |
| new_zh | SERVE_FAILED | - | - | - |
| old_zh | 1.00tok/round (0.0%)  | 2 | stop_token | 31.1tok/s |

ckpt = /mnt/c/Users/User/Documents/ziqinzhang/data/dflash2_ckpts/step_001200.pt (W9, step 1200); base = qwen3_8_27b_nvfp4.ninfer; new = qwen3_8_27b_nvfp4_dflash2_w9s1200.ninfer
读法: old vs new 的差值就是**草稿权重**的贡献; 若两者相近, 则瓶颈在机制而非训练。
目标 (用户口径): 位置1 接受率 ~0.9; 当前 dflash2 每轮 4.55/7 = 50.9%。
