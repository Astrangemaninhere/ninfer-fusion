
## width 判别：偏离点是否随 verify 的 T 变化 (2026-09-10 19:11)

plain 基准: 48 tokens
- mtp_d1  T=2 : **DIFFER at token 36** base=98633 spec=133222  
- mtp_d3  T=4 : **DIFFER at token 36** base=98633 spec=133222  
- mtp_d5  T=6 : **DIFFER at token 36** base=98633 spec=133222  

判读：三者若在**同一 token** 且错到**同一值** ⇒ verify 与 plain 之间有与 T 无关的固定差异（不是 batching 数值噪声）；
若偏离点/错值随 T 变化 ⇒ 属于 T 相关的数值差（kernel 实例化）。
