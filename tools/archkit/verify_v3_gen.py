# -*- coding: utf-8 -*-
"""gen_variant 语法验证 v3: body 置全局, 输入全局预声明。"""
import subprocess
import sys
import tempfile

sys.path.insert(0, '/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit')
import flavors
import gen_variant as gv

STUBS = '''
#include <initializer_list>
struct Tensor { Tensor slice(int,int,int) const; Tensor view(std::initializer_list<int>) const; };
struct Weight { int n, k; };
struct DType { enum : int { BF16 = 0, I32 = 1 }; };
struct Work { Tensor alloc(int, std::initializer_list<int>) { return {}; } };
namespace ops {
void attn_input_proj(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&, int, Work&, void*);
void linear(const Tensor&, const Weight&, Tensor&, void*);
void linear_add(const Tensor&, const Weight&, Tensor&, int, Work&, void*);
void rmsnorm(const Tensor&, const Weight&, float, bool, Tensor&, void*);
void rope(const Tensor&, int, float, Tensor&, Tensor&, void*);
void gqa_attention(const Tensor&, const Tensor&, Tensor&, const Tensor&, Tensor, const Tensor&, float, Tensor, Tensor, Work&, Tensor&, void*);
void sigmoid_mul(const Tensor&, Tensor&, void*);
void silu_mul(const Tensor&, const Tensor&, Tensor&, void*);
}
struct TextConfig { static constexpr int hidden=5120, intermediate=17408;
                    static constexpr float rms_epsilon=1e-6f; static constexpr int rotary_dim=64;
                    static constexpr float rope_theta=1e7f; };
static constexpr float kAttnScale = 1.0f;
int text_policy(const Weight&);
// ---- 全局输入 ----
Tensor x, q_flat, gate_flat, k_flat, v_flat, q, k, v, qn, kn, a, a_flat, act, pos;
Tensor valid, kv_table, kv_view, envelope, gate_up, g, u, gate;
Work work;
void* s;
static constexpr int T = 16;
Weight fused_qkgv;
Weight *q_w, *k_w, *v_w, *gate_up_w, *gate_w, *up_w, *down_w;
struct WW {
    Weight* input_norm = nullptr; Weight* q_norm = nullptr; Weight* k_norm = nullptr;
    Weight* o_proj = nullptr; Weight* post_attn_norm = nullptr;
    Weight* gate_up = nullptr; Weight* gate = nullptr; Weight* up = nullptr;
    Weight* down = nullptr; Weight* projection = nullptr;
};
WW w;
'''

G38 = {'hidden': 5120, 'query_heads': 24, 'kv_heads': 4, 'head_dim': 256,
       'vocab': 248320, 'max_ctx': 131072, 'intermediate': 17408}
G2 = {'hidden': 3584, 'query_heads': 28, 'kv_heads': 4, 'head_dim': 128,
      'vocab': 152064, 'max_ctx': 131072, 'intermediate': 18944}
for name, fam, emit in [('qwen38', 'qwen38', 'attn'), ('qwen38', 'qwen38', 'mlp'),
                        ('qwen2', 'qwen2', 'attn'), ('qwen2', 'qwen2', 'mlp')]:
    g = G38 if fam == 'qwen38' else G2
    pat = flavors.FLAVORS[fam]
    body = gv.emit_attention_leaf(g, pat.attn, name) if emit == 'attn' \
        else gv.emit_mlp_leaf(g, pat.mlp)
    hdr = '#pragma once\n' + STUBS + '\nvoid leaf() {\n' + body + '\n}\n'
    with tempfile.NamedTemporaryFile('w', suffix='.h', delete=False) as f:
        f.write(hdr)
        p = f.name
    r = subprocess.run(['g++', '-fsyntax-only', '-std=c++20', '-x', 'c++', p],
                       capture_output=True, text=True)
    print('%s-%s: %s' % (name, emit, 'SYNTAX-OK' if r.returncode == 0 else 'ERR'))
    if r.returncode != 0:
        print((r.stderr or '')[:350])
