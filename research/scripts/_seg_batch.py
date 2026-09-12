p = '/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/text_context_impl.h'
s = open(p).read()
old = '''        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, stream);
        ops::linear(hidden, *lm_head_, logits, stream);
    }
    work_.reset();'''
new = '''        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, batch});
        const int segment  = active_graph_segment_;
        const int segments = active_graph_segments_;
        if (segments <= 1) {
            ops::embedding(ids, *embed_, x, stream);
            NullTap tap;
            run_layers(x, Phase::Verify, tap);
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, stream);
            ops::linear(hidden, *lm_head_, logits, stream);
        } else {
            const int layer_total = kCfg.n_layers;
            const int layer_begin = segment * layer_total / segments;
            const int layer_end   = (segment + 1) * layer_total / segments;
            if (segment == 0) {
                ops::embedding(ids, *embed_, x, stream);
            }
            if (layer_begin < layer_end) {
                ScopedValue<std::int32_t> first_binding(active_layer_first_, layer_begin);
                ScopedValue<std::int32_t> last_binding(active_layer_last_, layer_end - 1);
                NullTap tap;
                run_layers(x, Phase::Verify, tap);
            }
            if (segment + 1 == segments) {
                ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, stream);
                ops::linear(hidden, *lm_head_, logits, stream);
            }
        }
    }
    work_.reset();'''
assert old in s
s = s.replace(old, new, 1)
open(p, 'w').write(s)
print('ordinary_decode_batch segmented')
